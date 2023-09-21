import json, os
from datasets import load_dataset

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
home_dir = os.path.expanduser('~')
data_dir = os.path.join(THIS_DIR, 'datasets')
if not os.path.exists(data_dir):
    os.mkdir(data_dir)

with open('datasets.json', 'r') as f:
    dataset_d = json.load(f)
for name in dataset_d:
    if '/' in name:
        nix_name = name.replace('/', '_')
    else:
        nix_name = name
    name_dir = os.path.join(data_dir, nix_name)
    if not os.path.exists(name_dir):
        os.mkdir(name_dir)
    d = dataset_d[name]
    if isinstance(d, list):
        subsets = d
        for subset in subsets:
            if name in ['bible_para', 'tatoeba', 'tilde_model', 'php', 'opus_wikipedia', 'qed_amara', 'open_subtitles', 'opus_ubuntu', 'opus_gnome', 'europa_eac_tm', 'kde4']:
                lang1, lang2 = subset['lang1'], subset['lang2']
                filename = '{}-{}'.format(lang1, lang2)
                data_path = os.path.join(name_dir, '{}.tfds'.format(filename))
            elif name == 'ted_talks_iwslt':
                lang1, lang2, year = subset['lang1'], subset['lang2'], subset['year']
                filename = '{}-{}-{}'.format(lang1, lang2, year)
                data_path = os.path.join(name_dir, '{}.tfds'.format(filename))
            else:
                data_path = os.path.join(name_dir, '{}.tfds'.format(subset))
            if not os.path.exists(data_path):
                try:
                    if name in ['bible_para', 'tatoeba', 'tilde_model', 'php' 'opus_wikipedia', 'qed_amara', 'open_subtitles', 'opus_ubuntu', 'opus_gnome', 'kde4']:
                        lang1, lang2 = subset['lang1'], subset['lang2']
                        dataset = load_dataset(name, lang1=lang1, lang2=lang2)
                    elif name == 'ted_talks_iwslt':
                        lang1, lang2, year = subset['lang1'], subset['lang2'], subset['year']
                        dataset = load_dataset(name, language_pair=(lang1, lang2), year=year)
                    elif name == 'europa_eac_tm':
                        lang1, lang2 = subset['lang1'], subset['lang2']
                        dataset = load_dataset(name, language_pair=(lang1, lang2))
                    else:
                        dataset = load_dataset(name, subset)
                except:
                    print('Error loading dataset {} {}'.format(name, subset))
                    continue
                if name == 'common_voice' or name == 'covost2':
                    cols = list(dataset.features)
                    cols.remove('audio')
                    dataset = dataset.select_columns(cols)
                dataset = dataset.with_format('tf')
                dataset.save_to_disk(data_path)
            else:
                print('Dataset {} already exists'.format(data_path))
    elif isinstance(d, dict):
        subsets = d.keys()
        print(subsets)
        for subset in subsets:
            splits = d[subset]
            print(splits)
            data_path = os.path.join(name_dir, '{}.tfds'.format(subset))
            if not os.path.exists(data_path):
                try:
                    print(name, subset, splits)
                    dataset = load_dataset(name, subset, split=splits)
                except:
                    print('Error loading dataset {} {} {}'.format(name, subset, splits))
                    continue
                dataset = dataset.with_format('tf')
                dataset.save_to_disk(data_path)
            else:
                print('Dataset {} already exists'.format(data_path))
    elif d == None:
        data_path = os.path.join(name_dir, '{}.tfds'.format(nix_name))
        if not os.path.exists(data_path):
            try:
                dataset = load_dataset(name)
            except:
                print('Error loading dataset {}'.format(name))
                continue
            dataset = dataset.with_format('tf')
            dataset.save_to_disk(data_path)
        else:
            print('Dataset {} already exists'.format(data_path))

if os.path.exists(os.path.join(home_dir, '.cache/huggingface/datasets')):
    os.system('rm -rf ~/.cache/huggingface/datasets')
