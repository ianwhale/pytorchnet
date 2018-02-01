import pickle
import os
import datetime
from config import parser

def main():
    args = parser.parse_args()
    # in case no gpu is available

    # 1, store the genome information
    with open("input_file/input%d.pkl" % int(args.genome_id - 1), "rb") as f:
        genome = pickle.load(f)
    if os.path.exists("error_evaluation"):
        pass
    else:
        os.makedirs("error_evaluation")
    filename = os.path.join("error_evaluation",
                            (datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')+".pkl"))
    with open(filename,"wb") as f:
        pickle.dump(genome,f)

    # 2, artificially assign a bad classification accuracy to it.
    accuracy = 0

    with open("output_file/output%d.pkl" % int(args.genome_id - 1), "wb") as f:
        pickle.dump(accuracy, f)

if __name__ == "__main__":
    main()