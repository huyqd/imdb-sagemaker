import argparse
import logging
import os
import sys

import tensorflow as tf
from datasets import load_dataset
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
import pandas as pd


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--learning_rate", type=str, default=5e-5)
    parser.add_argument("--do_train", type=bool, default=True) 
    parser.add_argument("--do_eval", type=bool, default=True)

    # Data, model, and output directories
    parser.add_argument("--output_data_dir", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"])
    parser.add_argument("--model_dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--n_gpus", type=str, default=os.environ["SM_NUM_GPUS"])
    parser.add_argument("--train", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    parser.add_argument("--test", type=str, default=os.environ["SM_CHANNEL_TEST"])

    args, _ = parser.parse_known_args()

    # Set up logging
    logger = logging.getLogger(__name__)

    logging.basicConfig(
        level=logging.getLevelName("INFO"),
        handlers=[logging.StreamHandler(sys.stdout)],
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Load model and tokenizer
    model = TFAutoModelForSequenceClassification.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Load dataset
    train = pd.read_csv(os.path.join(args.train, 'train.csv'), nrows=10)
    test = pd.read_csv(os.path.join(args.test, 'test.csv'), nrows=10)

    train_features = dict(tokenizer(train['text'].tolist(), truncation=True, padding="max_length", return_tensors='tf'))
    train_labels = tf.convert_to_tensor(train['label'], dtype=tf.int64)

    test_features = dict(tokenizer(test['text'].tolist(), truncation=True, padding="max_length", return_tensors='tf'))
    test_labels = tf.convert_to_tensor(test['label'], dtype=tf.int64)

    tf_train_dataset = tf.data.Dataset.from_tensor_slices((train_features, train_labels)).batch(
        args.train_batch_size
    )
    tf_test_dataset = tf.data.Dataset.from_tensor_slices((test_features, test_labels)).batch(
        args.eval_batch_size
    )

    # fine optimizer and loss
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metrics = [tf.keras.metrics.SparseCategoricalAccuracy()]
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    # Training
    if args.do_train:

        train_results = model.fit(tf_train_dataset, epochs=args.epochs, batch_size=args.train_batch_size)
        logger.info("*** Train ***")

        output_eval_file = os.path.join(args.output_data_dir, "train_results.txt")

        with open(output_eval_file, "w") as writer:
            logger.info("***** Train results *****")
            logger.info(train_results)
            for key, value in train_results.history.items():
                logger.info("  %s = %s", key, value)
                writer.write("%s = %s\n" % (key, value))

    # Evaluation
    if args.do_eval:

        result = model.evaluate(tf_test_dataset, batch_size=args.eval_batch_size, return_dict=True)
        logger.info("*** Evaluate ***")

        output_eval_file = os.path.join(args.output_data_dir, "eval_results.txt")

        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            logger.info(result)
            for key, value in result.items():
                logger.info("  %s = %s", key, value)
                writer.write("%s = %s\n" % (key, value))

    # Save result
    model.save_pretrained(args.model_dir)
    tokenizer.save_pretrained(args.model_dir)
