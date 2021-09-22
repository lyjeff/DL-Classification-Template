import os
import subprocess
from train import train
from eval import eval
from utils import argument_setting

if __name__ == '__main__':

    parser = argument_setting(inhert=True)
    parser.add_argument('--kaggle', type=str, metavar='Kaggle_Submission_Message', help="the submission message to upload Kaggle.")
    args = parser.parse_args()

    train(args)
    eval(args)

    print(f"Save output at {args.output_path}")

    if args.kaggle != None:
        print("Uploading Kaggle...")

        subprocess.run([
            "kaggle",
            "competitions",
            "submit",
            "-c"
            "dogs-vs-cats-redux-kernels-edition",
            "-f",
            os.path.join(args.output_path, "answer.csv"),
            "-m",
            args.kaggle
        ])

        print("All done!")
