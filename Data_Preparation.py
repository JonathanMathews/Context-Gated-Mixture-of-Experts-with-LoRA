import datasets

# 'image' and 'label'
game = datasets.load_dataset('Bingsu/Gameplay_Images')['train'].select(range(4000))

ttvGame = game.train_test_split(test_size = 0.2,
                                       shuffle = True,
                                       stratify_by_column = 'label')

tvGame = ttvGame['train'].train_test_split(test_size = 0.25,
                                           shuffle = True,
                                           stratify_by_column = 'label')

trainGame = tvGame['train']
valGame = tvGame['test']
testGame = ttvGame['test']

gameDataset = datasets.DatasetDict({'train': trainGame,
                                        'validation': valGame,
                                        'test': testGame})

gameDataset.save_to_disk('./game_dataset')

# 'image' and 'label'
fire = datasets.load_dataset("blanchon/FireRisk")['train'].select(range(43577))

zero = fire.select(range(1000))
one = fire.select(range(6296, 6296+1000))
two = fire.select(range(17001, 17001+1000))
three = fire.select(range(25618, 25618+1000))

reducedFire = datasets.concatenate_datasets([zero, one, two, three])

ttvFire = reducedFire.train_test_split(test_size = 0.2,
                                shuffle = True,
                                stratify_by_column = 'label')

tvFire = ttvFire['train'].train_test_split(test_size = 0.25,
                                                       shuffle = True,
                                                       stratify_by_column = 'label')

trainFire = tvFire['train']
valFire = tvFire['test']
testFire = ttvFire['test']

fireDataset = datasets.DatasetDict({'train': trainFire,
                                   'validation': valFire,
                                   'test': testFire})

fireDataset.save_to_disk('./fire_dataset')