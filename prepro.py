# coding: utf-8
from data_load import load_dataset
from hparame import Hparame
from bert import tokenization
import run_classifier_custom
import logging
from sklearn.model_selection import StratifiedKFold

logging.getLogger().setLevel(logging.INFO)


def create_tokenizer_from_hub_module(hp):
    """
    create tokenizer
    :return:
    """
    tokenization.validate_case_matches_checkpoint(True, hp.BERT_INIT_CHKPNT)

    return tokenization.FullTokenizer(vocab_file=hp.BERT_VOCAB, do_lower_case=True)


def process_data(hp):
    tokenizer = create_tokenizer_from_hub_module(hp)

    train = load_dataset('./input_data/train_process.csv')
    test = load_dataset('./input_data/test.csv')
    # train = train.sample(5000)
    # test = test.sample(5000)

    train = train[train['polarity'].isin([1, 2])]
    train.reset_index(inplace=True)
    sfolder = StratifiedKFold(n_splits=5, random_state=2020, shuffle=True)
    for kfold_num, (train_idx, eval_idx) in enumerate(sfolder.split(train[hp.DATA_COLUMN], train[hp.polarity])):
        # Use the InputExample class from BERT's run_classifier code to create examples from the data
        train_InputExamples = train.loc[train_idx].apply(lambda x:
                                                     run_classifier_custom.InputExample(guid=None,
                                                                                        # Globally unique ID for
                                                                                        # bookkeeping,
                                                                                        # unused in this example
                                                                                        text_a=x[hp.DATA_COLUMN],
                                                                                        selected_text=x[
                                                                                            hp.selected_text],
                                                                                        text_b=x[hp.sentiment],
                                                                                        sentiment=x[hp.polarity]),
                                                     axis=1)

        eval_InputExamples = train.loc[eval_idx].apply(lambda x:
                                                   run_classifier_custom.InputExample(guid=None,
                                                                                      text_a=x[hp.DATA_COLUMN],
                                                                                      selected_text=x[hp.selected_text],
                                                                                      text_b=x[hp.sentiment],
                                                                                      sentiment=x[hp.polarity]), axis=1)
        break

    # print(tokenizer.tokenize("This here's an example of using the BERT tokenizer"))

    # Convert our train and test features to InputFeatures that BERT understands.
    train_features = run_classifier_custom.convert_examples_to_features(train_InputExamples, hp.MAX_SEQ_LENGTH,
                                                                        tokenizer, is_predicting=False)
    eavl_features = run_classifier_custom.convert_examples_to_features(eval_InputExamples, hp.MAX_SEQ_LENGTH, tokenizer,
                                                                       is_predicting=False)

    return train_features, eavl_features


def process_test_data(hp):
    tokenizer = create_tokenizer_from_hub_module(hp)
    test = load_dataset('./input_data/test.csv')
    test_InputExamples = test.apply(lambda x: run_classifier_custom.InputExample(guid=None,
                                                                                 text_a=x[hp.DATA_COLUMN],
                                                                                 selected_text=None,
                                                                                 text_b=x[hp.sentiment],
                                                                                 sentiment=x[hp.polarity]), axis=1)
    test_features = run_classifier_custom.convert_examples_to_features(test_InputExamples, hp.MAX_SEQ_LENGTH, tokenizer,
                                                                       is_predicting=True)
    return test_features


if __name__ == "__main__":
    hparame = Hparame()
    parser = hparame.parser
    hp = parser.parse_args()

    train_features, test_features = process_data(hp)
