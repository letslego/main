# Penn Treebank

**FOR RESEARCH AND EDUCATIONAL PURPOSES ONLY**

We have access to the Penn Treebank through Berkeley, but it can be a pain to 
request it through the corpora site. Please do not use this distribution outside 
of the class.

To set this up so NLTK can find it:
```
cd $course_repo_dir
NLTK_DATA_DIR=$(python -c 'import nltk; print nltk.data.path[0]')
cp -r assignment2/data/ptb/* $NLTK_DATA_DIR/corpora/ptb
```

You can verify that it's installed with:
```
python -c 'from nltk.corpus import ptb; print ptb.fileids()'
```

You should see a list of IDs like `BROWN/CF/CF01.MRG`
