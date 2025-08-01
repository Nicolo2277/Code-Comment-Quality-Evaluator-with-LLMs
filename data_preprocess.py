import pandas as pd
import json
import random
from datasets import load_dataset
import ast
import re
from pathlib import Path
import logging
import cfg 

#Logging setup:
logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger(__name__)


class CodeSearchNetPreprocessor:
    '''Class to handle the loading and the preprocessing of the CodeSearchNet (python) dataset'''
    def __init__(self, sample_size=1000, min_comment_length=10, max_code_length=500):
        self.sample_size = sample_size
        self.min_comment_length = min_comment_length
        self.max_code_length = max_code_length
    
    def load_and_filter_dataset(self):
        '''We will load the dataset with only python-related code (and corresponding comments)'''
        logging.info('Loading CodeSearchNet Python dataset from huggingfaces')
        dataset = load_dataset('AhmedSSoliman/CodeSearchNet')
        logging.info(f"Loaded {len(dataset['train'])} Python examples")
        return dataset['train']
    
    def clean_code(self, code):
        '''Clean and validate Python code'''
        if not code:
            return None
        
        #Remove excessive whitespacing
        code = re.sub(r'\n\s*\n\s*\n', '\n\n', code)
        code = code.strip()

        code = "".join(ch for ch in code if ch.isprintable() or ch == '\n')
        code = re.sub(r'[ \t]+', ' ', code)
        code = re.sub(r'\n\n+', '\n\n', code)

        return code
    
    def clean_comment(self, comment):
        '''Clean and validate comments'''
        if not comment:
            return None
        
        comment = comment.strip().strip('"').strip("'")
        comment = re.sub(r'\s+', ' ', comment)

        if len(comment) < self.min_comment_length:
            return None
        
        return comment
    
    def is_a_good_example(self, code, comment):
        '''Filter criteria for good code-comment pairs'''
        if len(code) > self.max_code_length:
            return False
        
        if not code:
            return False
        
        if not comment:
            return False
        
        if 'def ' not in code:
            return False
        
        if len(comment.split()) < 3:
            return False
        
        return True
    
    def process_dataset(self):
        '''Main process pipeline'''
        dataset = self.load_and_filter_dataset()
        logging.info('Preprocessing and filtering the examples')
        processed_examples = []

        for i, example in enumerate(dataset):
            if i % 1000 == 0:
                logging.info(f'Processed {i} examples, found {len(processed_examples)} good ones')

            #Extract and clean data:
            code = self.clean_code(example.get('code', ''))
            comment = self.clean_comment(example.get('docstring', ''))

            if not code:
                continue

            if not comment:
                continue

            if not self.is_a_good_example(code, comment):
                continue

            processed_examples.append({
                'id': len(processed_examples),
                'code':code,
                'comment': comment,
                'clarity': None, #How clear is the comment?
                'usefulness': None, #How useful for understanding codes?
                'accuracy': None, #How accurate it is?
                'overall': None #Overall quality
            })

            if len(processed_examples) >= self.sample_size:
                break
        
        logging.info(f'Found {len(processed_examples)} good pairs')
        return processed_examples
    
    def save_data(self, data, filename):
        '''Save the data to json file'''
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        logging.info(f'Data saved succesfully in {filename}')

def main():

    args = cfg.parse_args()

    processor = CodeSearchNetPreprocessor(sample_size=args.sample_size, min_comment_length=args.min_comment_length,
                                          max_code_length=args.max_code_length)
    data = processor.process_dataset()
    processor.save_data(data=data, filename=args.data_file)
    logging.info('\nDataset preparation is complete')

if __name__ == '__main__':
    main()
    

