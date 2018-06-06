import unittest
import os
import sys
from io import StringIO
import data.transform_ud as t
import pandas as pd
import numpy as np
from functools import reduce
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config import config

import logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(module)s - %(funcName)s: %(message)s',
                    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)

INPUT = "datasets/MorphoData-NewSplit/dev.txt"
cols = list(config["DATASET"]["COLUMNS"].values())
defaults = config["TRANSFORM"]["DEFAULTS"]

args_run = []

for char_n_gram_param in [1, 2, 3]:
    args_run.append({'context_unit': 'char', 'char_n_gram': char_n_gram_param, 'context_size': None, 'context_char_size': 25})

for context_size_param in [2, 5, 10]:
    args_run.append({'context_unit': 'word', 'char_n_gram': None, 'context_size': context_size_param, 'context_char_size': None})

for context_char_size_param in [5, 10, 25]:
    args_run.append({'context_unit': 'char', 'char_n_gram': 2, 'context_size': None, 'context_char_size': context_char_size_param})


class TestTransformUD(unittest.TestCase):
    def setUp(self):
        self.wordlist = pd.read_csv(INPUT, sep='\s+', names=cols,
                                skip_blank_lines=False, comment='#')["word"].dropna().str.lower().unique()
        self.longMessage = True

    def generate_args(self, args):
        """
        Generates arguments to pass to transform_ud from a dict
        :param args:
        :return:
        """
        argv = []
        argv.append('--input')
        argv.append(INPUT)
        for k, v in args.items():
            if v:
                argv.append("--{}".format(k))
                argv.append(str(v))
        argv.append("--test_case")
        return argv

    def merge_into_words(self, list):
        """
        Converts a list of tag boundaries and subword units into a list of words
        :param list:
        :return:
        """
        res = []
        el = ""
        for sym in list:
            if "<" not in sym:
                el += sym.lower()
            elif sym == defaults["WORD_BOUNDARY"]:
                res.append(el)
                el = ""

        if el:
            res.append(el)

        return res

    def check_if_in_wordlist(self, units):
        """
        Checks if all elements in units are to be found among the unique lowercased wordforms in the INPUT dataset
        :param units:
        :return:
        """
        if units.size > 0:
            units_bool_mask = np.in1d(units, self.wordlist)
            self.assertIs(units_bool_mask.all(), np.bool_(True), "{} not in expected word list.".format(units[~units_bool_mask]))

    def get_number_of_units(self, units):
        """
        Counts units without tags
        :param units:
        :return: number of units without tags
        """
        return len([unit for unit in units if "<" not in unit])

    def get_number_of_chars(self, units):
        """
        Count the number of chars (excluding tags) in a list of subword units
        :param units:
        :return:
        """
        units.insert(0, 0)
        res = reduce((lambda x, y: x + len(y) if "<" not in y else x + 0), units)
        del units[0]
        return res

    def test_main(self):
            for i, args in enumerate(args_run):
                saved_stdout = sys.stdout
                argv = self.generate_args(args)
                with self.subTest(i=i):
                    try:
                        logger.info('Started ' + str(argv))
                        out = StringIO()
                        sys.stdout = out
                        t.main(argv)
                        out.seek(0)
                        sys.stdout = saved_stdout
                        logger.info("Output processed.")
                        for num_line, line in enumerate(out):
                            if defaults["RIGHT_CONTEXT_BOUNDARY"] not in line \
                                    or defaults["EXAMPLE_BOUNDARY"] not in line\
                                    or defaults["LEFT_CONTEXT_BOUNDARY"] not in line:
                                continue

                            lc = line.split(defaults["LEFT_CONTEXT_BOUNDARY"])[0]
                            rc = line.split(defaults["RIGHT_CONTEXT_BOUNDARY"])[1]

                            lc_units_and_tags = lc.split()
                            rc_units_and_tags = rc.split()

                            if args['context_size']:
                                lc_num = self.get_number_of_units(lc_units_and_tags)
                                rc_num = self.get_number_of_units(rc_units_and_tags)
                                self.assertTrue(lc_num + rc_num <= args['context_size'] * 2,
                                                "Context is too large for sentence: {}. Should be of size {} units.".format(line, args['context_size']))
                            elif args['context_char_size']:
                                lc_num = self.get_number_of_chars(lc_units_and_tags)
                                rc_num = self.get_number_of_chars(rc_units_and_tags)
                                self.assertTrue(lc_num + rc_num <= args['context_char_size'] * 2,
                                                "Context is too large for sentence: {}. Should be of size {} chars.".format(line, args['context_char_size']))

                            is_word = True if args['context_unit'] == 'word' else False

                            lc_words = np.array(self.merge_into_words(lc_units_and_tags))
                            rc_words = np.array(self.merge_into_words(rc_units_and_tags))

                            if lc_words.size > 0 and not is_word:
                                lc_words = np.delete(lc_words, 0)

                            if rc_words.size > 0 and not is_word:
                                rc_words = np.delete(rc_words, -1)

                            self.check_if_in_wordlist(lc_words)
                            self.check_if_in_wordlist(rc_words)

                            logger.info("Checks {}. {} ({}, {}) ({}, {})".format(num_line, line, lc_words, rc_words, lc_num, rc_num))
                        out.close()
                    except AssertionError:
                        logging.error("Assertion error logged: ", exc_info=True)
                    finally:
                        sys.stdout = saved_stdout


if __name__ == '__main__':
    unittest.main()