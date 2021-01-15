import random
import shutil
import os
from collections import defaultdict

class TestsGenerator:
    """
        Generates a test set by creating hard links to existing form images in the original dataset.
        Form images are NOT copied.
    """
    def __init__(
        self, metadata_path, forms_path,
        expected_output_path='correct_writers.txt',
        test_set_path='data', test_form_name='test',
        forms_extension='.png', test_samples_per_writer=2, writers_per_test=3):

        self.writers_forms = defaultdict(lambda: [])
        self.read_metadata(metadata_path)
        self.forms_path = forms_path

        self.expected_output_path = expected_output_path
        self.test_set_path = test_set_path
        self.test_form_name = test_form_name
        self.forms_extension = forms_extension
        self.test_samples_per_writer = test_samples_per_writer
        self.writers_per_test = writers_per_test

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
        testable_writers_count = 1 + random.choice(range(self.writers_per_test))
        non_testable_writers_count = self.writers_per_test - testable_writers_count

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

            if writer == correct_writer:
                # Write the writer index to the expected output file.
                with open(self.expected_output_path, 'a') as expected_output_file:
                    expected_output_file.write(str(writer_idx) + '\n')

                # Sample an extra form of the correct writer to use as the test form.
                sample_forms = random.sample(self.writers_forms[writer], test_samples_per_writer + 1)
                test_form = sample_forms.pop()

                # Create the test form hard link.
                src = self.forms_path + '/' + test_form + forms_extension
                dst = test_case_path + '/' + self.test_form_name + forms_extension
                os.link(src, dst)
            else:
                sample_forms = random.sample(self.writers_forms[writer], test_samples_per_writer)

            # Create hard links of the writer's sample forms.
            for form_idx, sample_form in enumerate(sample_forms, 1):
                src = self.forms_path + '/' + sample_form + forms_extension
                dst = test_writer_path + '/' + str(form_idx) + forms_extension
                os.link(src, dst)

    def generate_test_set(self, size):
        # Remove the test set directory if it already exists.
        shutil.rmtree(self.test_set_path, ignore_errors=True)
        os.mkdir(self.test_set_path)
        # Remove the expected output file if it already exists.
        try:
            os.remove(self.expected_output_path)
        except OSError:
            pass
        # Generate test cases.
        for test_case_idx in range(1, size + 1):
            test_case_path = self.test_set_path + '/' + f'{test_case_idx:02}'
            os.mkdir(test_case_path)
            self.generate_test_case(test_case_path)
