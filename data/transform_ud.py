#!/usr/bin/env python
import sys
import os
import pandas as pd
import re
from itertools import accumulate
from bisect import bisect
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config import config
defaults = config["TRANSFORM"]["DEFAULTS"]
cols = list(config["DATASET"]["COLUMNS"].values())


class Transformer(object):
    def __init__(self, word_unit=defaults["WORD_UNIT"],
                 tag_unit=defaults["TAG_UNIT"],
                 context_size=defaults["CONTEXT_SIZE"],
                 context_char_size=None,
                 left_context_boundary=defaults["LEFT_CONTEXT_BOUNDARY"],
                 right_context_boundary=defaults["RIGHT_CONTEXT_BOUNDARY"],
                 word_boundary=defaults["WORD_BOUNDARY"],
                 example_boundary=defaults["EXAMPLE_BOUNDARY"],
                 tag_boundary=defaults["TAG_BOUNDARY"],
                 subword_separator=defaults["SUBWORD_SEPARATOR"]):
        self.tag_unit = tag_unit
        self.tag_boundary = tag_boundary
        self.example_boundary = example_boundary
        self.subword_separator = subword_separator
        self.word_boundary = word_boundary
        self.left_context_boundary = left_context_boundary
        self.right_context_boundary = right_context_boundary
        self.context_size = context_size
        self.word_unit = word_unit
        self.context_char_size = context_char_size
        self.keep_punct_in_context = False

        # compute closing tag for each training example
        if self.example_boundary is not None:
            pos_close_tag = self.example_boundary.find('<') + 1
            self.close_tag = self.example_boundary[:pos_close_tag] + '/' + self.example_boundary[pos_close_tag:]

    def process_sentence(self, sentence_df, lc_df=pd.DataFrame(), rc_df=pd.DataFrame()):
        """
        Transforms a dataframe sentence into lines for all of the the sentence's words, each surrounded by its context.
        :param sentence_df: Input sentence dataframe, each row is a word (columns: word, lemma, tag)
        :param lc_df: Same format, but represents additional context to the left of the input sentence
        :param rc_df: Ditto
        :return: tuple(output_source_lines, output_target_lines)
        """
        # merges the sentence to be processed with its left and right context sentences (if any)
        sentence_with_context_df = pd.concat([lc_df, sentence_df, rc_df], copy=False)

        # check if subword units are present in the sentence and update flag accordingly
        self.subword_mode_flag = sentence_with_context_df["word"].str.contains(self.subword_separator).any()

        # by default removes all punctuation
        if not self.keep_punct_in_context:
            sentence_with_context_df = sentence_with_context_df[sentence_with_context_df["tag"] != "punct"]
            sentence_df = sentence_df[sentence_df["tag"] != "punct"]

        # lowercase proper nouns and other names
        proper_nouns_and_names = sentence_with_context_df["tag"].str.contains("^(?:Np|H).*$")
        sentence_with_context_df.loc[~proper_nouns_and_names, ["word"]] = \
            sentence_with_context_df.loc[~proper_nouns_and_names, ["word"]].squeeze(axis=1).str.lower()

        # remember the bounds of the sentence we want to process so that we are able to distinguish it from the
        # additional context
        sentence_start_idx = sentence_df.index[0]
        sentence_end_idx = sentence_df.index[-1]

        source_lines = []
        target_lines = []

        # this section calculated the source and target lines for each word in the sentence
        for index, row in sentence_with_context_df.loc[sentence_start_idx:sentence_end_idx + 1].iterrows():
            # we want to ignore subword separation for the word for which we calculate context
            # because it is always represent either wholly or on character-level
            wordform_clean = row["word"].replace(self.subword_separator + " ", "")

            source_line = []

            if self.example_boundary is not None:
                source_line.append(self.example_boundary)

            # computes left-hand side context
            source_line.extend(self.compute_context(sentence_with_context_df.loc[:index - 1]))

            source_line.append(self.left_context_boundary)

            # represent the word on word- or character-level
            if self.word_unit == 'word':
                source_line.append(wordform_clean)
            else:
                source_line.extend(wordform_clean)

            source_line.append(self.right_context_boundary)

            # computes right-hand side context
            # we have to revert the input and output in order to use the same function for both contexts
            rhs_sentence = sentence_with_context_df.loc[:index + 1:-1]

            # special preprocessing when breaking up words into subwords
            if self.subword_mode_flag:
                rhs_sentence = rhs_sentence.applymap(lambda x: x[::-1] if type(x) is str else x)

            right_context = self.compute_context(rhs_sentence)[::-1]

            # special postprocessing when breaking up words into subwords
            if self.subword_mode_flag:
                right_context_buffer = [w[::-1] if w is not self.word_boundary else w for w in right_context]
                right_context = right_context_buffer

            source_line.extend(right_context)

            if self.example_boundary is not None:
                source_line.append(self.close_tag)

            source_lines.append(" ".join(source_line))

            # computes target line
            # TODO: append moprho tags when requested
            # represent the word on word- or character-level
            target_line = []

            if self.example_boundary is not None:
                target_line.append(self.example_boundary)

            target_lemma = row["lemma"]
            target_tag = row["tag"]

            if self.word_unit == 'word':
                target_line.append(target_lemma)
            else:
                target_line.extend(target_lemma)

            target_line.append(self.tag_boundary)

            if self.tag_unit == 'word':
                target_line.append(target_tag)
            else:
                target_line.extend(target_tag)

            if self.example_boundary is not None:
                target_line.append(self.close_tag)

            target_lines.append(" ".join(target_line))

        return source_lines, target_lines

    def compute_context(self, context_df):
        """
        Computes context for a word based on constraints.
        :param context_df: dataframe representing left- or right-hand side of a sentence right before or after a
        word for which we want to compute context
        :return: a list of context units
        """
        context_units = []

        # split words if you find subword separators and append word boundaries in-between
        for word in context_df["word"].tolist():
            for subword in re.split("\s*{}\s*".format(self.subword_separator), word):
                context_units.append(subword)
            context_units.append(self.word_boundary)

        if self.context_char_size is not None:
            # code to return the desired initial number of characters from the context regardless of units
            cum_context_length = list(accumulate([len(unit) if unit is not self.word_boundary else 0
                                                  for unit in context_units[::-1]]))
            offset = bisect(cum_context_length, self.context_char_size)
            if offset != 0:
                context_units = context_units[-offset:]
            else:
                context_units = []
        else:
            # return only the initial number of units from the context
            context_units_buffer = []
            j = self.context_size
            for unit in context_units[::-1]:
                if unit != self.word_boundary:
                    j -= 1
                context_units_buffer.insert(0, unit)
                if j == 0:
                    break
            context_units = context_units_buffer

        # trims extraneous word boundary symbols from the output
        if context_units:
            if context_units[-1] is self.word_boundary:
                del context_units[-1]
        if context_units:
            if context_units[0] is self.word_boundary:
                del context_units[0]

        return context_units


def main(argv):
    import argparse
    from io import StringIO

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="adapts an UD dataset to context-sensitive lemmatization")

    parser.add_argument("--input", help="file to be transformed", type=str)
    parser.add_argument("--word_column_index", help="index of word column in the file (zero-indexed)", type=int,
                        default=0)
    parser.add_argument("--lemma_column_index", help="index of lemma column in the file (zero-indexed)", type=int,
                        default=1)
    parser.add_argument("--tag_column_index", help="index of tag column in the file (zero-indexed)", type=int,
                        default=2)
    parser.add_argument("--output", help="output source and target files", nargs='+', type=str, default=None)
    parser.add_argument("--context_unit", help="type of context representation", choices=['char', 'bpe', 'word'],
                        type=str, default=defaults["CONTEXT_UNIT"])
    parser.add_argument("--word_unit", help="type of word representation", choices=['char', 'word'],
                        type=str, default=defaults["WORD_UNIT"])
    parser.add_argument("--tag_unit", help="type of tag representation", choices=['char', 'word'],
                        type=str, default=defaults["TAG_UNIT"])
    parser.add_argument("--char_n_gram",
                        help="size of char-n-gram context (only used if --context_unit is char, default: %(default)s)",
                        type=int, default=defaults["CHAR_N_GRAM"])
    parser.add_argument("--context_size",
                        help="size of context representation (in respective units) on left and right (0 to use full span)",
                        type=int, default=defaults["CONTEXT_SIZE"])
    parser.add_argument("--context_char_size",
                        help="size of context representation (in characters) on left and right (0 to use full span, has precedence over --context_size)",
                        type=int, default=argparse.SUPPRESS)
    parser.add_argument("--context_span",
                        help="maximum span of a word in number of sentences on left and right of the sentence of the word, default: %(default)s))",
                        type=int,
                        default=defaults["CONTEXT_SPAN"])
    parser.add_argument("--left_context_boundary", help="left context boundary special symbol (default: %(default)s)",
                        type=str,
                        default=defaults["LEFT_CONTEXT_BOUNDARY"])
    parser.add_argument("--example_boundary", help="example boundary special symbol (default: %(default)s)", type=str,
                        default=defaults["EXAMPLE_BOUNDARY"])
    parser.add_argument("--right_context_boundary", help="right context boundary special symbol (default: %(default)s)",
                        type=str,
                        default=defaults["RIGHT_CONTEXT_BOUNDARY"])
    parser.add_argument("--word_boundary", help="word boundary special symbol (default: %(default)s)", type=str,
                        default=defaults["WORD_BOUNDARY"])
    parser.add_argument("--tag_boundary", help="tag boundary special symbol (default: %(default)s)", type=str,
                        default=defaults["TAG_BOUNDARY"])
    parser.add_argument('--subword_separator', type=str, default=defaults["SUBWORD_SEPARATOR"], metavar='STR',
                        help="separator between non-final BPE subword units (default: '%(default)s'))")
    parser.add_argument('--test_case', dest='test_case', action='store_true')
    parser.add_argument('--overwrite', dest='overwrite', action='store_true')
    parser.add_argument("--print_file", help="which file to output (source/target)", choices=['source', 'target'],
                        type=str, default=defaults["PRINT_FILE"])

    args = parser.parse_args(argv)

    if args.input is None:
        args.input = sys.stdin

    if not args.test_case:
        if args.output is None:
            if args.input is sys.stdin:
                raise ValueError(
                    "Can't decide how to name the transformation because you feed from stdin. Use --output to specify path.")

            input_folders = re.split("/+|\\+", args.input)
            if len(input_folders) < 2:
                raise ValueError("Can't decide how to name the transformation. Use --output to specify path.")
            else:
                input_folder = input_folders[-2]
                input_filename = input_folders[-1].split(".")[0]

            transform_folder = "{}_{}_{}{}".format(input_folder, args.context_size, args.context_unit,
                                                   "_{}".format(args.char_n_gram) if args.context_unit == "char" else "")

            full_transform_folder_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'input', transform_folder)
            os.makedirs(full_transform_folder_path, exist_ok=True)

            output_source_path = os.path.join(full_transform_folder_path, '{}_source'.format(input_filename))
            output_target_path = os.path.join(full_transform_folder_path, '{}_target'.format(input_filename))

            if not args.overwrite and (os.path.isfile(output_source_path) or os.path.isfile(output_target_path)):
                raise ValueError("Output files for {} already exist in {}. Pass --overwrite or delete them."
                                 .format(input_filename, full_transform_folder_path))

            # truncate output files or create them anew
            open(output_source_path, 'w').close()
            open(output_target_path, 'w').close()
        else:
            if len(args.output) < 2:
                raise ValueError("You must specify both target and source output paths.")
            output_source_path = args.output[0]
            output_target_path = args.output[1]

    transformer_args = {'word_unit': args.word_unit, 'tag_unit': args.tag_unit, 'context_size': args.context_size,
                        'context_char_size': args.context_char_size if hasattr(args, 'context_char_size') else None,
                        'left_context_boundary': args.left_context_boundary, 'tag_boundary': args.tag_boundary,
                        'right_context_boundary': args.right_context_boundary, 'word_boundary': args.word_boundary,
                        'example_boundary': args.example_boundary, 'subword_separator': args.subword_separator}

    transformer = Transformer(**transformer_args)

    # preprocessing
    with open(args.input, 'r', encoding='utf-8') as infile:
        infile_buffer = StringIO()
        prev_line = ""
        empty_line = ['\n', '\r\n']
        for i, line in enumerate(infile):
            if line.startswith("\""):
                continue
            if line in empty_line and prev_line in empty_line:
                continue
            infile_buffer.write(line)
            prev_line = line
        infile_buffer.seek(0)

    infile_df = pd.read_csv(infile_buffer, sep='\s+', names=cols,
                            usecols=[args.word_column_index, args.lemma_column_index, args.tag_column_index],
                            skip_blank_lines=False, comment='#')[cols]

    if args.context_unit == 'char':
        from types import SimpleNamespace
        import numpy as np
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'subword-nmt'))
        from subword_nmt.segment_char_ngrams import segment_char_ngrams
        subword_nmt_output = StringIO()
        segment_char_ngrams(SimpleNamespace(input=infile_df["word"].dropna().astype(str), vocab={}, n=args.char_n_gram,
                                            output=subword_nmt_output, separator=args.subword_separator))
        subword_nmt_output.seek(0)
        infile_df.loc[infile_df["word"].notnull(), ["word"]] = np.array([line.rstrip(' \t\n\r')
                                                                         for line in subword_nmt_output])[:, np.newaxis]
    elif args.context_unit == 'bpe':
        # TODO: BPE processing should happen here
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'subword-nmt'))
        from subword_nmt.learn_bpe import learn_bpe

    sentence_dfs = []
    sentence_indices = pd.isna(infile_df).all(axis=1)
    sentence_iterator = (i for i, e in sentence_indices.to_dict().items() if e is True)

    sentence_start = 0
    for sentence_end in sentence_iterator:
        sentence_dfs.append(infile_df.iloc[sentence_start:sentence_end])
        sentence_start = sentence_end + 1

    for sentence_df in sentence_dfs:
        # TODO: add additional context according to CONTEXT_SPAN to line before passing it below
        output_source_lines, output_target_lines = transformer.process_sentence(sentence_df)

        if args.test_case:
            if args.print_file == 'source':
                print("\n".join(output_source_lines))
            else:
                print("\n".join(output_target_lines))
        else:
            with open(output_source_path, 'a+', encoding='utf-8') as outsourcefile, \
                    open(output_target_path, 'a+', encoding='utf-8') as outtargetfile:
                outsourcefile.write("\n".join(output_source_lines) + "\n")
                outtargetfile.write("\n".join(output_target_lines) + "\n")


if __name__ == "__main__":
    main(sys.argv[1:])
