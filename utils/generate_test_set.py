import argparse
import random
import shutil
import os
from collections import defaultdict
import json

expected_output_path = 'correct_writers.txt'
map_path = 'map.json'
test_set_path = 'data'
test_form_name = 'test'
forms_extension = '.png'
test_samples_per_writer = 2
writers_per_test = 3


class TestsGenerator:
    """
        Generates a test set by creating hard links to existing form images in the original dataset.
        Form images are NOT copied.
    """
    def __init__(self, metadata_path, forms_path, map_path):
        self.writers_forms = defaultdict(lambda: [])
        self.read_metadata(metadata_path)
        self.forms_path = forms_path
        self.map = None
        if map_path is not None:
            with open(map_path, 'r') as f:
                self.map = json.load(f)
        self.mymap = {'dirs': [], 'links': [], 'expected_output': ''}

    def read_metadata(self, metadata_file_path):
        # Get the form ids for each writer.
        with open(metadata_file_path) as metadata_file:
            for line in metadata_file:
                # Ignore comments.
                if line.startswith('#'):
                    continue
                form_id, writer_id, _ = line.split(maxsplit=2)
                self.writers_forms[writer_id].append(form_id)

        # Filter out writer with no enough forms to construct a test case.
        self.writers_forms = {writer: forms for writer, forms in self.writers_forms.items()
                              if len(forms) >= test_samples_per_writer}

    def pick_test_writers(self):
        # Testable writers are writers that have enough forms for test case samples + 1 test form.
        non_testable_writers = \
            [writer for writer, forms in self.writers_forms.items() if len(forms) < test_samples_per_writer + 1]
        testable_writers = \
            [writer for writer, forms in self.writers_forms.items() if len(forms) >= test_samples_per_writer + 1]

        # Decide how many writers to pick out of each group, we need at least one testable writer.
        testable_writers_count = 1 + random.choice(range(writers_per_test))
        non_testable_writers_count = writers_per_test - testable_writers_count

        writers = random.sample(testable_writers, testable_writers_count) + \
            random.sample(non_testable_writers, non_testable_writers_count)

        # Before shuffling the first writer always belongs to the testable writers.
        correct_writer = writers[0]
        random.shuffle(writers)
        return writers, correct_writer

    def generate_test_case(self, test_case_path):
        writers, correct_writer = self.pick_test_writers()
        for writer_idx, writer in enumerate(writers, 1):
            # Create a directory for the writer's sample forms.
            test_writer_path = test_case_path + '/' + str(writer_idx)
            os.mkdir(test_writer_path)
            self.mymap['dirs'].append(test_writer_path)

            if writer == correct_writer:
                # Write the writer index to the expected output file.
                with open(expected_output_path, 'a') as expected_output_file:
                    expected_output_file.write(str(writer_idx) + '\n')

                # Sample an extra form of the correct writer to use as the test form.
                sample_forms = random.sample(self.writers_forms[writer], test_samples_per_writer + 1)
                test_form = sample_forms.pop()

                # Create the test form hard link.
                src = self.forms_path + '/' + test_form + forms_extension
                dst = test_case_path + '/' + test_form_name + forms_extension
                os.link(src, dst)

                self.mymap['links'].append([os.path.relpath(self.forms_path, src), dst])
            else:
                sample_forms = random.sample(self.writers_forms[writer], test_samples_per_writer)

            # Create hard links of the writer's sample forms.
            for form_idx, sample_form in enumerate(sample_forms, 1):
                src = self.forms_path + '/' + sample_form + forms_extension
                dst = test_writer_path + '/' + str(form_idx) + forms_extension
                os.link(src, dst)

                self.mymap['links'].append([os.path.relpath(self.forms_path, src), dst])

    def generate_test_set(self, size):
        # Remove the test set directory if it already exists.
        shutil.rmtree(test_set_path, ignore_errors=True)
        os.mkdir(test_set_path)
        self.mymap['dirs'].append(test_set_path)
        # Remove the expected output file if it already exists.
        try:
            os.remove(expected_output_path)
        except OSError:
            pass
        if self.map is None:
            # Generate test cases.
            for test_case_idx in range(1, size + 1):
                test_case_path = test_set_path + '/' + f'{test_case_idx:02}'
                os.mkdir(test_case_path)
                self.mymap['dirs'].append(test_case_path)
                self.generate_test_case(test_case_path)
        else:
            # Link ready generated cases.
            for dir in self.map['dirs']:
                os.makedirs(dir, exist_ok=True)
            for pair in self.map['links']:
                src = os.path.join(self.forms_path, pair[0])
                dst = pair[1]
                os.link(src, dst)
            with open(expected_output_path, 'w') as f:
                f.write(self.map['expected_output'])
            self.mymap = self.map
        
        # Save the mapping for later.
        with open(map_path, 'w') as f:
            with open(expected_output_path) as f2:
                 self.map['expected_output'] = f2.read()
            json.dump(self.mymap, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'Create a test set from a given dataset. '
        'Test forms are generated by creating hard links to existing dataset form files')
    parser.add_argument('--forms', required=True,
                        help='The path of the directory containing the dataset forms.')
    parser.add_argument('--size', type=int, required=True,
                        help='The number of test cases to generate.')
    parser.add_argument('--metadata', default='dataset_forms_metadata.txt',
                        help='The path of the dataset forms metadata file.')
    parser.add_argument('--map', default=None,
                        help='The path to map.json file to rebuild the same data')

    args = parser.parse_args()
    TestsGenerator(args.metadata, args.forms, args.map).generate_test_set(args.size)
