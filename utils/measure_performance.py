import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Measures the performance of the writer identifier.')
    parser.add_argument('--results', default='results.txt',
                        help='The path of the results file produced by the writer identifier.')
    parser.add_argument('--expected_output', default='correct_writers.txt',
                        help='The path of the expected outputs file of the test set.')
    parser.add_argument('--time', default='time.txt',
                        help='The path of the time file produced by the writer identifier.')
    args = parser.parse_args()

    with open(args.results) as results_file:
        results = results_file.readlines()

    with open(args.expected_output) as expected_output_file:
        correct_writers = expected_output_file.readlines()

    assert len(results) == len(correct_writers), \
        'The # of produced results must be equal to the number of expected outputs'

    correct_identifications = 0
    for result, expected_output in zip(results, correct_writers):
        correct_identifications += int(result) == int(expected_output)

    accuracy = correct_identifications / len(results)

    with open(args.time) as time_file:
        times = time_file.readlines()

    average_time = sum([int(time) for time in times]) / len(times)

    print('Accuracy: {}, Average time: {}'.format(accuracy, average_time))
