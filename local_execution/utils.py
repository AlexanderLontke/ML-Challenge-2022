import csv
import torch

from constants import classes_to_label


def create_submission(net, submission_dataloader, filename: str = "submission.csv"):
    """
    Helper method which creates a Kaggle submission from a given model and
    :param net:
    :param submission_dataloader:
    :param filename:
    :return:
    """
    submission_results = []

    index = 0
    with torch.no_grad():
        for images in iter(submission_dataloader):
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            label = classes_to_label[int(predicted[0])]
            # Print predicted sample every once in a while
            if index % 1000 == 0:
                print(f"Predicted: {label}")
            submission_results.append([index, label])
            index += 1

    # field names
    fields = ["test_id", "label"]

    # writing to csv file
    with open(filename, "w") as csvfile:
        # creating a csv writer object
        csv_writer = csv.writer(csvfile)

        # writing the fields
        csv_writer.writerow(fields)

        # writing the data rows
        csv_writer.writerows(submission_results)
    print(f"Submission was written to ./{filename}")
