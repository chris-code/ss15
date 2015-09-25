# Brainstorming: libraries


## Scientific computation

The most important libraries (at least for us) are [NumPy](http://www.numpy.org/) and [SciPy](http://www.scipy.org/). They contain the most important mathematical methods, like: 

* linear algebra
* statistics
* signal processing
* optimization
* Fourier Transforms
* ...

There are many others, specialised for specific fields. For instance [Natural Language Toolkit (NLTK)](http://www.nltk.org/) for linguistic tasks.

And then visualization is important, of course. The most popular library is [matplotlib](http://matplotlib.org/).

## Machine learning

You can find many different ML problems here:

* [Kaggle](https://www.kaggle.com/)
* [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets.html)

* https://www.kaggle.com/c/predict-closed-questions-on-stack-overflow
* https://www.kaggle.com/c/galaxy-zoo-the-galaxy-challenge
* https://www.kaggle.com/c/higgs-boson
* https://www.kaggle.com/c/denoising-dirty-documents
* https://www.kaggle.com/c/emc-data-science
* https://www.kaggle.com/c/crowdflower-search-relevance
* https://www.kaggle.com/c/sf-crime
* https://www.kaggle.com/c/job-recommendation
* https://www.kaggle.com/c/avito-prohibited-content
* https://www.kaggle.com/c/avito-context-ad-clicks

Some useful libraries containing different ML methods:

* [scikit-learn](http://scikit-learn.org/stable/)
* [Modular Toolkit for Data Processing (MDP)](http://mdp-toolkit.sourceforge.net/): It's not very actively developed anymore but we still use it a lot in our workgroup.

## (Social) network analysis

There are several Python libraries for (social) network analysis like [NetworkX](http://networkx.github.io/) or [graph-tool](https://graph-tool.skewed.de/). Datasets containing networks can found for instance here:

* [Stanford Large Network Dataset Collection](http://snap.stanford.edu/data/index.html)
* [Gephi Wiki](https://github.com/gephi/gephi/wiki/Datasets)
* [Social Graphs in Movies](http://moviegalaxies.com/)

## Scraping websites

* [import.io](https://import.io/): Service that extracts data from websites
* [BeautifulSoup](http://www.crummy.com/software/BeautifulSoup/): Convenient access to content of a downloaded website
* [Scrapy](http://scrapy.org/): Framework for scraping websites
* [Selenium](http://www.seleniumhq.org/): Allows complete automation of a browser via script

Think of data sources like concert tickets or products, movies (IMDB)...

Note however, that many websites don't need to be scraped because they offer a proper API to access their content. Here are examples from a [long list](http://www.programmableweb.com/category/all/apis?order=field_popularity) with some of the most popular web APIs:

* Google Maps, Twitter, YouTube, Flickr, Facebook, Amazon Product Advertising, Twillo, Last.fm, eBay, ...


# Examples


## Google's dreaming neural networks

Google realeased the Python scripts for it's famous dreaming neural networks ([deepdream](https://github.com/google/deepdream)). Others have build on that, for instance [making *Fear and Loathing in Las Vegas* even more uncanny](https://github.com/graphific/DeepDreamVideo):

![Fear and Loathing example](https://camo.githubusercontent.com/dcf15823a576975a5bd2d1af1696a25a07b7e6aa/687474703a2f2f6d656469612e67697068792e636f6d2f6d656469612f6c34316c537a6a5473474a63497a704b672f67697068792e676966)

Here's another project that created a simple user interface: [bat-country](https://github.com/jrosebr1/bat-country)

## Implementing Machine learning algorithms on your own

Basically all you need is numpy and the corresponding math.

Examples: Feed Forward Neural Networks, Auto encoders, Sparse Coding, Baysian Networks, Restricted Boltzmann machines, ...

## Solving a machine learning problem

Using existing libs like scikit the focus would be on solving a particular task in a good way.
So it would be problem driven and you should have a concrete idea about the problem

Have a look at http://scikit-learn.org/stable/index.html for examples for clustering, resgression, classification, ...

Examples: Object detection (Eye, hand, ...), Spam detection, ...