import numpy as np
import itertools
from collections import Counter
import os
from nltk.tokenize import sent_tokenize
import string
from pymystem3 import Mystem
import re
import zipfile as zip
import rarfile as rar
import time
import shutil
lemmat = Mystem()
rar.UNRAR_TOOL=r"C:\Program Files\WinRAR\UnRAR.exe"

N = 5
MAX_LENGTH = 80

slash = "/"

classes = [['HISTORY', 'INOSTRHIST'],
           ['DETEKTIWY', 'RUSS_DETEKTIW'],
           ['KIDS', 'TALES'],
           ['SONGS', 'POEZIQ'],
           ['SCIFICT', 'RUFANT', 'INOFANT', 'TOLKIEN']]
											
											
def to_one_hot(i, N):
    return [(1 if j == i else 0) for j in range(N)]


def load_data_and_labels():
    class_examples = [[] for i in range(N)]
    path = r"dataset_help"
    dirs = os.listdir(path)
    dirs = sorted(dirs)
    print(dirs)
    for i, dir in enumerate(dirs):
        for file in os.listdir(path + slash + dir):
            with open(str(path + slash + dir + slash + file), "r", encoding="cp1251") as f:
                 temp = []
                 for sen in f.read().split(sep="\n"):
                     if len(sen) < MAX_LENGTH:
                         temp.append(sen)
                 class_examples[i] += temp
    print("Loading data from files finished!")

    x = []
    for examples in class_examples:
        x += examples
    x = [s.split(" ") for s in x]
    print("Splitting by words finished!")

    class_labels = [[to_one_hot(i, N) for j in range(len(class_examples[i]))] for i in range(N)]
    y = np.concatenate(class_labels, 0)
    print("Generating labels finished!")

    return [x, y]


class ExtendSent:
    def __init__(self, senten, sep="<SEP/>", size = 0):
        self.sentenc = senten
        self.sep = sep
        self.size = max(len(x) for x in senten) if size == 0 else size
        print("Hello There")

    def __iter__(self):
        for s in self.sentenc:
            yield s + [self.sep] * (self.size - len(s))


def build_vocabulary(sentences):
    generator = ExtendSent(sentences)
    count = Counter(itertools.chain(*[[w for w in sent] for sent in generator]))
    vocabulary_inverse = [x[0] for x in count.most_common()]
    vocabulary_inverse = {key: value for key, value in enumerate(vocabulary_inverse)}
    vocabulary = {value: key for key, value in vocabulary_inverse.items()}
    print("Building vocabulary finished!")
    return [vocabulary, vocabulary_inverse]


def load_data():
    sentences, y = load_data_and_labels()
    vocabulary, vocabulary_inverse = build_vocabulary(sentences)
    pad_generator = ExtendSent(sentences)
    x = np.array([[vocabulary[word] for word in sentence] for sentence in pad_generator])
    y = np.array(y)
    return [x, y, vocabulary, vocabulary_inverse]


def create_classes_dataset():
    path_old = r"library"
    path_new = r"dataset"
    with open(r"files_path.txt", "r") as f:
        for s in f.readlines():
            s = s.replace(r"C:\Users\Андрей\Desktop\raw_data\book_1\\", "")
            head = s[:s.find(slash)]
            for i in range(N):
                if head in classes[i]:
                    s = s.split(slash)[-1].strip()
                    try:
                        shutil.copyfile(path_old + slash + s,
                                        path_new + slash + str(i) + slash + s + ".txt")
                    except:
                        print("File: " + s + " not found!")


def preparation_sample_file(file_path="", res_folder="", stopW_path=""):
    with open(r"stop_words.txt", "r", encoding="cp1251") as stop_w:
        stop_word = {i for i in stop_w.read().split()}

    pattern = r"[^а-яА-Я0-9.?!, ]+"
    file_path = r"test" + slash + "catarrh.txt"
    res_folder = r"test" + slash + "1"
    res = None
    with open(file_path, "r", encoding="cp1251") as inp, \
         open(res_folder + slash + file_path.split(slash)[-1], "w", encoding="cp1251") as outp:
        s = inp.read()
        text = None
        try:
            text = re.sub(pattern, " ", s)
            text = sent_tokenize(text, language='russian')
            result = []
            for sent in text:
                sentence = []
                for word in sent.lower().split():
                    word = word.strip(string.punctuation)
                    if word not in stop_word:
                        sentence.append(word)
                sentence.append("<s>")
                result.append(' '.join(sentence))
            rez = lemmat.lemmatize(''.join(result))
            rez = "".join(rez).replace("<s>", "\n")
            outp.write(rez)
        except UnicodeDecodeError:
            print("IT'S ERROR FILE", file_path)
    return rez


def give_me_paths(path):
    raw_d_dirs = set()
    for root, dirs, files in os.walk(path):
        print(root)
        for file in files:
            if file.endswith(".html"):
                raw_d_dirs.add(root + "\\" + file)
            elif zip.is_zipfile(root + "\\" + file):
                try:
                    with zip.ZipFile(root + "\\" + file, "r") as arch:
                        arch.extractall(root)
                        for f in arch.filelist:
                            if f.filename.endswith(".html"):
                                raw_d_dirs.add(root + "\\" +
                                               f.filename.replace("/", "\\"))
                except zip.BadZipfile:
                    print("Bad zip ->", root + "\\" + file)
                except:
                    print("Unknown error", root + "\\" + file)
            elif rar.is_rarfile(root + "\\" + file):
                try:
                    with rar.RarFile(root + "\\" + file, "r") as arch:
                        arch.extractall(root)
                        for f in arch.infolist():
                            if (f.filename.endswith(".html")):
                                raw_d_dirs.add(root + "\\" +
                                               f.filename.replace("/", "\\"))
                except rar.BadRarFile:
                    print("Bad rar ->", root + "\\" + file)
                except:
                    print("Unknown Error", root + "\\" + file)
    return raw_d_dirs


def normalize_text(files_path="", stop_w_path="", res_folder="", error_files="", final_paths=""):
    with open(r"tmp_path.txt", "r") as files:
        list_files = [file for file in files.read().split()]

    with open(r"stop_words.txt", "r") as stop_w:
        stop_word = {i for i in stop_w.read().split()}

    pattern = r"[^а-яА-Я0-9.?!, ]+"
    res_folder = r"NewDataset1"
    error_files, final_paths = [], []
    with open(r"error_files.txt", "w") as err:
        err.write("\n".join(error_files))
    for cntr, file in enumerate(list_files):
        st = time.time()
        outp_string = res_folder + slash + slash.join(file.split(slash)[1:])
        print(outp_string)
        with open(file, "r") as inp, \
                open(outp_string, "w") as outp:
            try:
                text = re.sub(pattern, " ", inp.read())
                final_paths.append(outp_string)
            except UnicodeDecodeError:
                print("IT'S ERROR FILE", file)
                error_files.append(file)
                continue
            text = sent_tokenize(text, language='russian')
            result = []
            for sent in text:
                sentence = []
                for word in sent.lower().split():
                    word = word.strip(string.punctuation)
                    if word not in stop_word:
                        sentence.append(word)
                sentence.append("<s>")
                result.append(' '.join(sentence))
            rez = lemmat.lemmatize(''.join(result))
            outp.write("".join(rez).replace("<s>", "\n"))
        if cntr % 40 == 0:
            with open(r"error_files.txt", "a") as err:
                err.write("\n".join(error_files))
                error_files.clear()
        print(time.time() - st)

    new_dir = r"final_paths_new.txt"
    with open(new_dir, "w") as res:
        res.write("\n".join(final_paths))