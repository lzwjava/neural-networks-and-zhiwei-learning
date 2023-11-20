import os


def main():
    dir_path = 'data/names'
    for name in os.listdir(dir_path):
        with open(os.path.join(dir_path, name)) as f:
            print(f.read())


if __name__ == '__main__':
    main()
