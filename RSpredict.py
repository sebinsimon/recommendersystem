from surprise import KNNBasic, SlopeOne
from surprise import Dataset

# Load the movielens-100k dataset
data = Dataset.load_builtin('ml-100k')

# Retrieve the trainset.
trainset = data.build_full_trainset()

# Build an algorithm, and train it.
algo = SlopeOne()
algo.fit(trainset)

uid = str(11)  # raw user id (as in the ratings file). They are **strings**!
iid = str(51)  # raw item id (as in the ratings file). They are **strings**!

# get a prediction for specific users and items.
pred = algo.predict(uid, iid, verbose=True)
