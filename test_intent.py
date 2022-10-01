from constants import INTENT_CKPT_DIRECTORY
from dataset import SeqClsTestDataset
from parse_test_args import parse_test_args
from testers import IntentTester

if __name__ == "__main__":
    args = parse_test_args("./data/intent/test.json", "./cache/intent/", INTENT_CKPT_DIRECTORY, "pred.intent.csv")
    tester = IntentTester(args.cache_dir / "intent2idx.json", SeqClsTestDataset, args.max_len, args.batch_size,
                          args.cache_dir, args.test_file, args.ckpt_dir, args.pred_file)
    tester.test()
