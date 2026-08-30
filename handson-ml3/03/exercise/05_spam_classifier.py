"""演習4: スパム分類器。

Apache SpamAssassinの公開コーパス(ham/spam)を取得し、メールをHTML除去・小文字化・
数字/URL置換・語幹抽出(stemming)を経て単語カウントベクトルに変換するパイプラインを
自作し、LogisticRegressionで分類する。
"""

import email
import email.parser
import email.policy
from pathlib import Path
import re
import sys
import tarfile
import urllib.request
from collections import Counter
from html import unescape

import numpy as np
from loguru import logger
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import precision_score, recall_score
from sklearn.pipeline import Pipeline

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from data_produce.util.logging_config import setup_logger

SPAM_ROOT = "http://spamassassin.apache.org/old/publiccorpus/"
HAM_URL = SPAM_ROOT + "20030228_easy_ham.tar.bz2"
SPAM_URL = SPAM_ROOT + "20030228_spam.tar.bz2"
SPAM_DATA_DIR = Path(__file__).resolve().parent / "data" / "spam"


def fetch_spam_data() -> tuple[Path, Path]:
    SPAM_DATA_DIR.mkdir(parents=True, exist_ok=True)
    for dir_name, tar_name, url in (
        ("easy_ham", "ham", HAM_URL),
        ("spam", "spam", SPAM_URL),
    ):
        if not (SPAM_DATA_DIR / dir_name).is_dir():
            path = (SPAM_DATA_DIR / tar_name).with_suffix(".tar.bz2")
            logger.info(f"{url} をダウンロードします")
            urllib.request.urlretrieve(url, path)
            with tarfile.open(path) as tar_bz2_file:
                tar_bz2_file.extractall(path=SPAM_DATA_DIR, filter="data")
        else:
            logger.info(f"{SPAM_DATA_DIR / dir_name} が既に存在するため、ダウンロードをスキップします")
    return SPAM_DATA_DIR / "easy_ham", SPAM_DATA_DIR / "spam"


def load_email(filepath: Path):
    with open(filepath, "rb") as f:
        return email.parser.BytesParser(policy=email.policy.default).parse(f)


def get_email_structure(mail) -> str:
    if isinstance(mail, str):
        return mail
    payload = mail.get_payload()
    if isinstance(payload, list):
        multipart = ", ".join(get_email_structure(sub_email) for sub_email in payload)
        return f"multipart({multipart})"
    return mail.get_content_type()


def structures_counter(emails) -> Counter:
    structures = Counter()
    for mail in emails:
        structures[get_email_structure(mail)] += 1
    return structures


def html_to_plain_text(html: str) -> str:
    text = re.sub("<head.*?>.*?</head>", "", html, flags=re.M | re.S | re.I)
    text = re.sub(r"<a\s.*?>", " HYPERLINK ", text, flags=re.M | re.S | re.I)
    text = re.sub("<.*?>", "", text, flags=re.M | re.S)
    text = re.sub(r"(\s*\n)+", "\n", text, flags=re.M | re.S)
    return unescape(text)


def email_to_text(mail) -> str | None:
    html = None
    for part in mail.walk():
        ctype = part.get_content_type()
        if ctype not in ("text/plain", "text/html"):
            continue
        try:
            content = part.get_content()
        except Exception:
            content = str(part.get_payload())
        if ctype == "text/plain":
            return content
        html = content
    if html:
        return html_to_plain_text(html)
    return None


class EmailToWordCounterTransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        strip_headers=True,
        lower_case=True,
        remove_punctuation=True,
        replace_urls=True,
        replace_numbers=True,
        stemming=True,
    ):
        self.strip_headers = strip_headers
        self.lower_case = lower_case
        self.remove_punctuation = remove_punctuation
        self.replace_urls = replace_urls
        self.replace_numbers = replace_numbers
        self.stemming = stemming

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        import nltk
        import urlextract

        stemmer = nltk.PorterStemmer() if self.stemming else None
        url_extractor = urlextract.URLExtract() if self.replace_urls else None

        X_transformed = []
        for mail in X:
            text = email_to_text(mail) or ""
            if self.lower_case:
                text = text.lower()
            if self.replace_urls and url_extractor is not None:
                urls = list(set(url_extractor.find_urls(text)))
                urls.sort(key=len, reverse=True)
                for url in urls:
                    text = text.replace(url, " URL ")
            if self.replace_numbers:
                text = re.sub(r"\d+(?:\.\d*)?(?:[eE][+-]?\d+)?", "NUMBER", text)
            if self.remove_punctuation:
                text = re.sub(r"\W+", " ", text, flags=re.M)
            word_counts = Counter(text.split())
            if self.stemming and stemmer is not None:
                stemmed_word_counts = Counter()
                for word, count in word_counts.items():
                    stemmed_word_counts[stemmer.stem(word)] += count
                word_counts = stemmed_word_counts
            X_transformed.append(word_counts)
        return np.array(X_transformed, dtype=object)


class WordCounterToVectorTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, vocabulary_size=1000):
        self.vocabulary_size = vocabulary_size

    def fit(self, X, y=None):
        total_count = Counter()
        for word_count in X:
            for word, count in word_count.items():
                total_count[word] += min(count, 10)
        most_common = total_count.most_common()[: self.vocabulary_size]
        self.vocabulary_ = {word: index + 1 for index, (word, count) in enumerate(most_common)}
        return self

    def transform(self, X, y=None):
        rows, cols, data = [], [], []
        for row, word_count in enumerate(X):
            for word, count in word_count.items():
                rows.append(row)
                cols.append(self.vocabulary_.get(word, 0))
                data.append(count)
        return csr_matrix((data, (rows, cols)), shape=(len(X), self.vocabulary_size + 1))


def main() -> None:
    setup_logger(Path(__file__).stem)

    ham_dir, spam_dir = fetch_spam_data()
    ham_filenames = [f for f in sorted(ham_dir.iterdir()) if len(f.name) > 20]
    spam_filenames = [f for f in sorted(spam_dir.iterdir()) if len(f.name) > 20]
    logger.info(f"ham: {len(ham_filenames)}件, spam: {len(spam_filenames)}件")

    ham_emails = [load_email(f) for f in ham_filenames]
    spam_emails = [load_email(f) for f in spam_filenames]

    logger.info(f"hamの構造内訳(上位): {structures_counter(ham_emails).most_common(5)}")
    logger.info(f"spamの構造内訳(上位): {structures_counter(spam_emails).most_common(5)}")
    logger.info(
        "hamはtext/plainが多く、spamはtext/htmlが多い傾向がある"
        "(メールの構造自体が有用な特徴になりうる)"
    )

    X = np.array(ham_emails + spam_emails, dtype=object)
    y = np.array([0] * len(ham_emails) + [1] * len(spam_emails))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    logger.info("=== 前処理パイプライン(メール→単語カウント→ベクトル)を先頭3件で確認 ===")
    X_few_wordcounts = EmailToWordCounterTransformer().fit_transform(X_train[:3])
    logger.info(f"単語カウント例: {X_few_wordcounts}")
    vocab_transformer = WordCounterToVectorTransformer(vocabulary_size=10)
    X_few_vectors = vocab_transformer.fit_transform(X_few_wordcounts)
    logger.info(f"ベクトル化例(先頭3件, 語彙10件):\n{X_few_vectors.toarray()}")
    logger.info(f"語彙: {vocab_transformer.vocabulary_}")

    logger.info("=== 訓練データ全体を前処理してLogisticRegressionで交差検証 ===")
    preprocess_pipeline = Pipeline([
        ("email_to_wordcount", EmailToWordCounterTransformer()),
        ("wordcount_to_vector", WordCounterToVectorTransformer()),
    ])
    X_train_transformed = preprocess_pipeline.fit_transform(X_train)

    log_clf = LogisticRegression(max_iter=1000, random_state=42)
    scores = cross_val_score(log_clf, X_train_transformed, y_train, cv=3)
    logger.info(f"3-fold交差検証Accuracy: mean={scores.mean():.4f}")

    logger.info("=== テストデータでの適合率・再現率 ===")
    X_test_transformed = preprocess_pipeline.transform(X_test)
    log_clf = LogisticRegression(max_iter=1000, random_state=42)
    log_clf.fit(X_train_transformed, y_train)
    y_pred = log_clf.predict(X_test_transformed)

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    logger.info(f"Precision: {precision:.2%}, Recall: {recall:.2%}")
    logger.info(
        "高い適合率・再現率の両方を達成できており、'easy_ham'の単純なデータセットでは"
        "単語出現ベースの特徴量とLogisticRegressionでも十分な性能が出ることを確認できた"
    )


if __name__ == "__main__":
    main()
