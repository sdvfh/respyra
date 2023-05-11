from os import path

SCRIPT_PATH = path.dirname(path.realpath(__file__))

FULL_CSV_PATH = path.join(SCRIPT_PATH, 'resources/full.csv')
GENERATE_CSV_PATH = path.join(SCRIPT_PATH, '/resources/generated_data.csv')




if __name__ == '__main__':
    print(FULL_CSV_PATH)