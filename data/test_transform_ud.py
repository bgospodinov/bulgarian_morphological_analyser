import unittest
import os
import sys
from io import StringIO
import data.transform_ud as t
import pandas as pd
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config import config

import logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

INPUT = "datasets/MorphoData-NewSplit/dev.txt"
cols = list(config["DATASET"]["COLUMNS"].values())
defaults = config["TRANSFORM"]["DEFAULTS"]

args_run = []
args_run.append({'context_unit': 'word', 'char_n_gram': None, 'context_size': 5, 'context_char_size': None})


class TestTransformUD(unittest.TestCase):
    def setUp(self):
        self.wordlist = pd.read_csv(INPUT, sep='\s+', names=cols,
                                skip_blank_lines=False, comment='#')["word"].dropna().str.lower().unique()
        self.longMessage = True

    def generate_args(self, args):
        argv = []
        argv.append('--input')
        argv.append(INPUT)
        for k, v in args.items():
            if v:
                argv.append("--{}".format(k))
                argv.append(str(v))
        argv.append("--test_case")
        return argv

    def cleanOutput(self, list):
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
        if units.size > 0:
            units_bool_mask = np.in1d(units, self.wordlist)
            self.assertTrue(units_bool_mask.all(), "{} not in expected word list.".format(units[~units_bool_mask]))

    def get_unit_size(self, units):
        pass

    def get_char_size(self, units):
        pass

    def test_main(self):
        saved_stdout = sys.stdout
        try:
            for i, args in enumerate(args_run):
                with self.subTest(i=i):
                    argv = self.generate_args(args)
                    logger.info('Started ' + str(argv))
                    out = StringIO()
                    sys.stdout = out
                    t.main(argv)
                    out.seek(0)
                    for line in out:
                        if defaults["RIGHT_CONTEXT_BOUNDARY"] not in line\
                                or defaults["LEFT_CONTEXT_BOUNDARY"] not in line:
                            continue

                        lc = line.split(defaults["LEFT_CONTEXT_BOUNDARY"])[0]
                        rc = line.split(defaults["RIGHT_CONTEXT_BOUNDARY"])[1]

                        lc_units = lc.split()
                        rc_units = rc.split()

                        if args['context_size']:
                            self.assertTrue(self.get_unit_size(lc_units) + self.get_unit_size(rc_units) <= args['context_size'] * 2)

                        lc_words = np.array(self.cleanOutput(lc_units))
                        rc_words = np.array(self.cleanOutput(rc_units))

                        self.check_if_in_wordlist(lc_words)
                        self.check_if_in_wordlist(rc_words)
        finally:
            sys.stdout = saved_stdout


if __name__ == '__main__':
    unittest.main()