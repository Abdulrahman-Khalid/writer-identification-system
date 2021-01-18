# Writer Identification System
Identify the writer of a handwritten document out of 3 writers.

## Requirements
- python3
- python3-pip
- virtualenv

```
$ ./setup.sh
```

## Run The Classifier
```
$ python3 identify_writer.py
```
The previous script will:
1. Iterate over `data` directory in the current directory.
2. Train the classifier over the training set for each patch.
3. Identify the writer for each patch.
4. Write output to `time.txt` and `results.txt` in the current directory.

For more options:
```
$ python3 identify_writer.py --help
```

## Generate Test Set
Register and download the [IAM Dataset](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database).

Then generate a test set from the dataset, use the test set generator utility:

```
$ python3 utils/generate_test_set.py
```

The dataset is generated by creating hard links to existing form images in the original dataset.
The form images are NOT copied.

Find out the different options using:

```
$ python3 utils/generate_test_set.py --help
```

By default, the script assumes the dataset forms metadata file exists in the root directory 
and is named `dataset_forms_metadata.txt`.

Example:

```
$ python3 utils/generate_test_set.py --forms iam_dataset --size 1000
```

## Measure Performance

To measure the writer identifier's performance, in terms of accuracy and speed, use the performance measurement utility:

```
$ python3 utils/measure_performance.py
```

Find out the different options using:

```
$ python3 utils/measure_performance.py --help
```

By default, the script assumes the results, time per test case, and expected output files exist in the root directory, 
and are named `results.txt`, `time.txt` and `correct_writers.txt` respectively.
