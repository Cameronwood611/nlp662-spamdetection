# nlp662-spamdetection



## Training and Test Datasets

 - We downloaded all datasets from https://spamassassin.apache.org/old/publiccorpus/

 - Suggested commands to unpack files:
 - The folders these create are placed in the gitignore and not pushed to the repo.

 
    + tar xvjf 20030228_easy_ham_2.tar.bz2
    + tar xvjf 20030228_easy_ham.tar.bz2
    + tar xvjf 20030228_hard_ham.tar.bz2
    + tar xvjf 20030228_spam.tar.bz2
    + tar xvjf 20050311_spam_2.tar.bz2


## Run Jupyter Notebook

 - The way to run this code is on cheaha inside jupyter notebook. Run each cell sequentially. All cells and functions are documented thoroughly.

## Run webap

 - We decided to save and load our model on a Flask backend and use ReactJS to upload files and show the results on the frontend.

 - If you'd like to run the webapp you need to have the Python package manager [pdm](https://pdm.fming.dev/), NodeJS package manager [npm](https://docs.npmjs.com/downloading-and-installing-node-js-and-npm), and [GNU Make](https://www.gnu.org/software/make/) to use the Makefile.

Type the following command:

`make init`

Now, in two separate terminal windows, type the following two commands:
 - `make start-backend`
 - `make start-frontend`

The webapp should now be accessible on localhost:3000 or http://127.0.0.1:3000 and you can upload emails as `.txt` files and the model will predict if it is SPAM or HAM.

## References

 - We leveraged this [article](https://towardsdatascience.com/spam-detection-in-emails-de0398ea3b48) for our SpamDetection model.

---

###### ETC.
authors: Cameron Wood (@cameronwood611), Tom el Safadi (@tsafadi)

tags: nlp, lstm, spam-detection