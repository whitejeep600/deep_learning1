from constants import SLOT_CKPT_DIRECTORY
from dataset import SeqTaggingClsTestDataset
from parse_test_args import parse_test_args
from testers import SlotTester

if __name__ == "__main__":
    args = parse_test_args("./data/slot/test.json", "./cache/slot/", SLOT_CKPT_DIRECTORY, "pred.slot.csv")
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    tester = SlotTester(args.cache_dir / "tag2idx.json", SeqTaggingClsTestDataset, args.max_len, args.batch_size,
                        args.cache_dir, args.test_file, args.ckpt_dir, args.pred_file)
    tester.test()
