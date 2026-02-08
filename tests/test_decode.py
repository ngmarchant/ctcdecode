"""Test decoders."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest
import torch
import ctcdecode
import os


class TestDecoders(unittest.TestCase):
    def setUp(self):
        self.vocab_list = ['\'', ' ', 'a', 'b', 'c', 'd', '_']
        self.beam_size = 20
        self.probs_seq1 = [[
            0.06390443, 0.21124858, 0.27323887, 0.06870235, 0.0361254,
            0.18184413, 0.16493624
        ], [
            0.03309247, 0.22866108, 0.24390638, 0.09699597, 0.31895462,
            0.0094893, 0.06890021
        ], [
            0.218104, 0.19992557, 0.18245131, 0.08503348, 0.14903535,
            0.08424043, 0.08120984
        ], [
            0.12094152, 0.19162472, 0.01473646, 0.28045061, 0.24246305,
            0.05206269, 0.09772094
        ], [
            0.1333387, 0.00550838, 0.00301669, 0.21745861, 0.20803985,
            0.41317442, 0.01946335
        ], [
            0.16468227, 0.1980699, 0.1906545, 0.18963251, 0.19860937,
            0.04377724, 0.01457421
        ]]
        self.probs_seq2 = [[
            0.08034842, 0.22671944, 0.05799633, 0.36814645, 0.11307441,
            0.04468023, 0.10903471
        ], [
            0.09742457, 0.12959763, 0.09435383, 0.21889204, 0.15113123,
            0.10219457, 0.20640612
        ], [
            0.45033529, 0.09091417, 0.15333208, 0.07939558, 0.08649316,
            0.12298585, 0.01654384
        ], [
            0.02512238, 0.22079203, 0.19664364, 0.11906379, 0.07816055,
            0.22538587, 0.13483174
        ], [
            0.17928453, 0.06065261, 0.41153005, 0.1172041, 0.11880313,
            0.07113197, 0.04139363
        ], [
            0.15882358, 0.1235788, 0.23376776, 0.20510435, 0.00279306,
            0.05294827, 0.22298418
        ]]
        self.greedy_result = ["ac'bdc", "b'da"]
        self.beam_search_result = ['acdc', "b'a", "a a"]

    def convert_to_string(self, tokens, vocab, seq_len):
        return ''.join([vocab[x] for x in tokens[0:seq_len]])

    def test_beam_search_decoder_group(self):
        probs_seq = torch.FloatTensor([self.probs_seq1])
        decoder = ctcdecode.CTCBeamDecoder(self.vocab_list, beam_width=self.beam_size,
                                           blank_id=self.vocab_list.index('_'),
                                           num_groups=2,
                                           diversity_factor=0.00001)
        beam_result, beam_scores, timesteps, out_seq_len = decoder.decode(probs_seq)
        output_str = self.convert_to_string(beam_result[0][0], self.vocab_list, out_seq_len[0][0])
        # print('\n'.join(
        #     self.convert_to_string(beam_result[0][i], self.vocab_list, out_seq_len[0][i])
        #     for i in range(len(beam_result[0]))
        # ))
        # self.assertEqual(len(set(output_str)), len(output_str))
        self.assertEqual(output_str, self.beam_search_result[0])

    def test_npsc(self):
        probs = torch.load('../../data/NPSC/ctc_test_am_probs.pth')[:2]
        decoder = ctcdecode.CTCBeamDecoder(
            labels=list('_abcdefghijklmnopqrstuvwxyzæøåéü '),
            beam_width=4,  # Need at least four beams, otherwise both groups will perform greedy search
            num_groups=2,
            diversity_factor=10,
            num_processes=1,
        )
        beam_result, beam_scores, timesteps, out_seq_len = decoder.decode(probs)
        self.assert_diversity_test(beam_result, decoder, out_seq_len)

    def test_npsc_lm(self):
        probs = torch.load('../../data/NPSC/ctc_test_am_probs.pth')[:2]
        decoder = ctcdecode.CTCBeamDecoder(
            labels=list('_abcdefghijklmnopqrstuvwxyzæøåéü '),
            beam_width=16,
            num_groups=2,
            diversity_factor=1e-20,  # The addition of the LM causes much smaller probabilities overall
            num_processes=1,
            alpha=1.3, beta=1.97,
            model_path='../../models/kenlm-npsc/lm_SESS_train_2021_03_02_114223.binary'
        )
        beam_result, beam_scores, timesteps, out_seq_len = decoder.decode(probs)
        self.assert_diversity_test(beam_result, decoder, out_seq_len)

    def assert_diversity_test(self, beam_result, decoder, out_seq_len):
        len_floats = torch.tensor(out_seq_len, dtype=torch.float)
        group_size = decoder._beam_width // decoder._num_groups
        group_1 = len_floats[:, :group_size]
        group_2 = len_floats[:, group_size:]
        print('1:', group_1.mean().item(), '±', group_1.std().item())
        print('2:', group_2.mean().item(), '±', group_2.std().item())
        for i in range(len(beam_result)):
            candidate1 = self.convert_to_string(beam_result[i][0], decoder._labels, out_seq_len[i][0])
            candidate2 = self.convert_to_string(beam_result[i][group_size], decoder._labels, out_seq_len[i][group_size])
            self.assertNotEqual(candidate1[:100], candidate2[:100])  # Ensure differences in early part of transcript
            print(candidate1, candidate2, sep='\n')
        # Ensure approx. equal lengths
        self.assertLessEqual(group_1.mean() - 2*group_1.std(), group_2.mean() + 2*group_2.std())

    def test_beam_search_decoder_1(self):
        probs_seq = torch.FloatTensor([self.probs_seq1])
        decoder = ctcdecode.CTCBeamDecoder(self.vocab_list, beam_width=self.beam_size,
                                           blank_id=self.vocab_list.index('_'))
        beam_result, beam_scores, timesteps, out_seq_len = decoder.decode(probs_seq)
        output_str = self.convert_to_string(beam_result[0][0], self.vocab_list, out_seq_len[0][0])
        self.assertEqual(output_str, self.beam_search_result[0])

    def test_beam_search_decoder_2(self):
        probs_seq = torch.FloatTensor([self.probs_seq2])
        decoder = ctcdecode.CTCBeamDecoder(self.vocab_list, beam_width=self.beam_size,
                                           blank_id=self.vocab_list.index('_'))
        beam_result, beam_scores, timesteps, out_seq_len = decoder.decode(probs_seq)
        output_str = self.convert_to_string(beam_result[0][0], self.vocab_list, out_seq_len[0][0])
        self.assertEqual(output_str, self.beam_search_result[1])

    def test_beam_search_decoder_3(self):
        lm_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'test.arpa')
        probs_seq = torch.FloatTensor([self.probs_seq2])

        decoder = ctcdecode.CTCBeamDecoder(self.vocab_list, beam_width=self.beam_size,
                                           blank_id=self.vocab_list.index('_'),
                                           model_path=lm_path, num_groups=2, diversity_factor=0.1)
        beam_result, beam_scores, timesteps, out_seq_len = decoder.decode(probs_seq)
        output_str = self.convert_to_string(beam_result[0][0], self.vocab_list, out_seq_len[0][0])
        self.assertEqual(output_str, self.beam_search_result[2])

    def test_beam_search_decoder_batch(self):
        probs_seq = torch.FloatTensor([self.probs_seq1, self.probs_seq2])
        decoder = ctcdecode.CTCBeamDecoder(self.vocab_list, beam_width=self.beam_size,
                                           blank_id=self.vocab_list.index('_'), num_processes=24)
        beam_results, beam_scores, timesteps, out_seq_len = decoder.decode(probs_seq)
        output_str1 = self.convert_to_string(beam_results[0][0], self.vocab_list, out_seq_len[0][0])
        output_str2 = self.convert_to_string(beam_results[1][0], self.vocab_list, out_seq_len[1][0])
        self.assertEqual(output_str1, self.beam_search_result[0])
        self.assertEqual(output_str2, self.beam_search_result[1])

    def test_beam_search_decoder_batch_log(self):
        probs_seq = torch.FloatTensor([self.probs_seq1, self.probs_seq2]).log()
        decoder = ctcdecode.CTCBeamDecoder(self.vocab_list, beam_width=self.beam_size,
                                           blank_id=self.vocab_list.index('_'), log_probs_input=True,
                                           num_processes=24)
        beam_results, beam_scores, timesteps, out_seq_len = decoder.decode(probs_seq)
        output_str1 = self.convert_to_string(beam_results[0][0], self.vocab_list, out_seq_len[0][0])
        output_str2 = self.convert_to_string(beam_results[1][0], self.vocab_list, out_seq_len[1][0])
        self.assertEqual(output_str1, self.beam_search_result[0])
        self.assertEqual(output_str2, self.beam_search_result[1])


if __name__ == '__main__':
    unittest.main()
