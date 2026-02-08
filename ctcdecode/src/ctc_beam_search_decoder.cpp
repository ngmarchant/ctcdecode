#include "ctc_beam_search_decoder.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <map>
#include <utility>

#include "decoder_utils.h"
#include "ThreadPool.h"
#include "fst/fstlib.h"
#include "path_trie.h"

using FSTMATCH = fst::SortedMatcher<fst::StdVectorFst>;

DecoderState::DecoderState(
    const std::vector<std::string> &vocabulary,
    size_t beam_size,
    double cutoff_prob,
    size_t cutoff_top_n,
    size_t blank_id,
    int log_input,
    Scorer *ext_scorer,
    size_t num_groups,
    float_t diversity_factor)
    : abs_time_step(0),
      beam_size(beam_size),
      cutoff_prob(cutoff_prob),
      cutoff_top_n(cutoff_top_n),
      blank_id(blank_id),
      log_input(log_input),
      vocabulary(vocabulary),
      ext_scorer(ext_scorer),
      num_groups(num_groups),
      diversity_factor(diversity_factor) {
  // assign space id
  auto it = std::find(vocabulary.begin(), vocabulary.end(), " ");
  // if no space in vocabulary
  if (it == vocabulary.end()) {
    space_id = -2;
  } else {
    space_id = std::distance(vocabulary.begin(), it);
  }

  group_size = beam_size / num_groups;

  // init prefixes' root
  for (size_t i = 0; i < num_groups; ++i) {
    auto *root = new PathTrie(i);
    roots.push_back(root);
    root->score = root->log_prob_b_prev = 0.0;

    if (ext_scorer != nullptr && !ext_scorer->is_character_based()) {
      auto *dictionary = static_cast<fst::StdVectorFst *>(ext_scorer->dictionary);
      root->set_dictionary(dictionary);
    }

    auto group = std::vector<PathTrie *>();
    group.push_back(root);
    groups.push_back(group);
  }
}

void DecoderState::next(const std::vector<std::vector<double>> &probs_seq) {
  // dimension check
  size_t num_time_steps = probs_seq.size();
  for (size_t i = 0; i < num_time_steps; ++i) {
    VALID_CHECK_EQ(probs_seq[i].size(),
                   vocabulary.size(),
                   "The shape of probs_seq does not match with "
                   "the shape of the vocabulary");
  }

  // prefix search over time
  for (size_t time_step = 0; time_step < num_time_steps; ++time_step, ++abs_time_step) {
    for (size_t group = 0; group < num_groups; ++group) {
      beam_step(probs_seq, time_step, group);
    }
  }  // end of loop over time
}

void DecoderState::beam_step(const std::vector<std::vector<double>> &probs_seq, size_t time_step, size_t group) {
  auto &prefixes = groups[group];
  auto &prob = probs_seq[time_step];

  float min_cutoff = -NUM_FLT_INF;
  bool full_beam = false;
  size_t num_prefixes = std::min(prefixes.size(), beam_size);
  if (ext_scorer != nullptr) {
    std::sort(prefixes.begin(), prefixes.begin() + num_prefixes, prefix_compare);
    float blank_prob = log_input ? prob[blank_id] : log(prob[blank_id]);
    min_cutoff = prefixes[num_prefixes - 1]->score + blank_prob - std::max(0.0, ext_scorer->beta) - diversity_factor;
    full_beam = (num_prefixes == group_size);
  }

  std::vector<std::pair<size_t, float>> log_prob_idx =
      get_pruned_log_probs(prob, cutoff_prob, cutoff_top_n, log_input);
  // loop over chars
  for (size_t index = 0; index < log_prob_idx.size(); index++) {
    auto c = log_prob_idx[index].first;
    auto log_prob_c = log_prob_idx[index].second;

    for (size_t i = 0; i < prefixes.size() && i < beam_size; ++i) {
      auto prefix = prefixes[i];
      if (full_beam && log_prob_c + prefix->score < min_cutoff) {
        break;
      }
      // blank
      if (c == blank_id) {
        prefix->log_prob_b_cur =
            log_sum_exp(prefix->log_prob_b_cur, log_prob_c + prefix->score);
        continue;
      }
      // repeated character
      if (c == prefix->character) {
        prefix->log_prob_nb_cur = log_sum_exp(
            prefix->log_prob_nb_cur, log_prob_c + prefix->log_prob_nb_prev);
      }
      // get new prefix
      auto prefix_new = prefix->get_path_trie(c, abs_time_step, log_prob_c);

      if (prefix_new != nullptr) {
        float log_p = -NUM_FLT_INF;

        if (c == prefix->character &&
            prefix->log_prob_b_prev > -NUM_FLT_INF) {
          log_p = log_prob_c + prefix->log_prob_b_prev;
        } else if (c != prefix->character) {
          log_p = log_prob_c + prefix->score;
        }

        // language model scoring
        if (ext_scorer != nullptr &&
            (c == space_id || ext_scorer->is_character_based())) {
          PathTrie *prefix_to_score;
          // skip scoring the space
          if (ext_scorer->is_character_based()) {
            prefix_to_score = prefix_new;
          } else {
            prefix_to_score = prefix;
          }

          std::vector<std::string> ngram = ext_scorer->make_ngram(prefix_to_score);
          float score = ext_scorer->get_log_cond_prob(ngram) * ext_scorer->alpha;
          log_p += score;
          log_p += ext_scorer->beta;
        }

        prefix_new->log_prob_nb_cur = log_sum_exp(prefix_new->log_prob_nb_cur, log_p);
      }

    }  // end of loop over prefix
  }    // end of loop over vocabulary

  // Add diversity bonus
  if (group > 0 && time_step > 0) {
    for (auto prefix : prefixes) {
      size_t distance = 0;
      for (size_t lower_group = 0; lower_group < group; ++lower_group) {
        for (auto other : groups[lower_group]) {
          if (abs((long) (other->length - prefix->length)) >= 10) continue;
          distance += prefix->character != other->character;
        }
      }
      prefix->diversity_bonus = log(distance * diversity_factor / group);
    }
  }

  prefixes.clear();
  // update log probs
  roots[group]->iterate_to_vec(prefixes);

  // only preserve top beam_size prefixes
  if (prefixes.size() >= group_size) {
    std::nth_element(prefixes.begin(),
                     prefixes.begin() + group_size,
                     prefixes.end(),
                     prefix_compare_diversity);
    for (size_t i = prefixes.size() - 1; i >= group_size; --i) {
      prefixes[i]->remove();
    }

    prefixes.resize(group_size);
  }
}

std::vector<std::pair<double, Output>>
DecoderState::decode() const {
  auto prefixes = std::vector<PathTrie *>();
  for (size_t i = 0; i < num_groups; ++i) {
    auto &group = groups[i];
    prefixes.insert(prefixes.end(), group.begin(), group.end());
  }
  std::unordered_map<const PathTrie *, float> scores;
  for (PathTrie *prefix : prefixes) {
    scores[prefix] = prefix->score;
  }

  // score the last word of each prefix that doesn't end with space
  if (ext_scorer != nullptr && !ext_scorer->is_character_based()) {
    for (size_t i = 0; i < beam_size && i < prefixes.size(); ++i) {
      auto prefix = prefixes[i];
      if (!prefix->is_empty() && prefix->character != space_id) {
        std::vector<std::string> ngram = ext_scorer->make_ngram(prefix);
        float score = ext_scorer->get_log_cond_prob(ngram) * ext_scorer->alpha;
        score += ext_scorer->beta;
        scores[prefix] += score;
      }
    }
  }

  using namespace std::placeholders;
  size_t num_prefixes = std::min(prefixes.size(), beam_size);
  std::sort(prefixes.begin(), prefixes.begin() + num_prefixes,
            std::bind(prefix_compare_external_scores, _1, _2, scores));

  return get_beam_search_result(prefixes, beam_size);
}

std::vector<std::pair<double, Output>> ctc_beam_search_decoder(
    const std::vector<std::vector<double>> &probs_seq,
    const std::vector<std::string> &vocabulary,
    size_t beam_size,
    double cutoff_prob,
    size_t cutoff_top_n,
    size_t blank_id,
    int log_input,
    Scorer *ext_scorer,
    size_t num_groups,
    float_t diversity_factor
) {
  DecoderState state(vocabulary, beam_size, cutoff_prob, cutoff_top_n, blank_id,
                     log_input, ext_scorer, num_groups, diversity_factor);
  state.next(probs_seq);
  return state.decode();
}


std::vector<std::vector<std::pair<double, Output>>>
ctc_beam_search_decoder_batch(
    const std::vector<std::vector<std::vector<double>>> &probs_split,
    const std::vector<std::string> &vocabulary,
    size_t beam_size,
    size_t num_processes,
    double cutoff_prob,
    size_t cutoff_top_n,
    size_t blank_id,
    int log_input,
    Scorer *ext_scorer,
    size_t num_groups,
    float_t diversity_factor
) {
  VALID_CHECK_GT(num_processes, 0, "num_processes must be nonnegative!");
  VALID_CHECK_GT(num_groups, 0, "num_groups must be at least 1!");
  if (beam_size % num_groups > 0) {
    printf(
        "beam_size=%zu should be divisible by num_groups=%zu! Try beam_size=%zu\n",
        beam_size, num_groups, std::max((size_t) (ceil(beam_size / num_groups) * num_groups), num_groups * 2)
    );
  }
  // thread pool
  ThreadPool pool(num_processes);
  // number of samples
  size_t batch_size = probs_split.size();

  // enqueue the tasks of decoding
  std::vector<std::future<std::vector<std::pair<double, Output>>>> res;
  for (size_t i = 0; i < batch_size; ++i) {
    res.emplace_back(pool.enqueue(ctc_beam_search_decoder,
                                  probs_split[i],
                                  vocabulary,
                                  beam_size,
                                  cutoff_prob,
                                  cutoff_top_n,
                                  blank_id,
                                  log_input,
                                  ext_scorer,
                                  num_groups,
                                  diversity_factor));
  }

  // get decoding results
  std::vector<std::vector<std::pair<double, Output>>> batch_results;
  for (size_t i = 0; i < batch_size; ++i) {
    batch_results.emplace_back(res[i].get());
  }
  return batch_results;
}
