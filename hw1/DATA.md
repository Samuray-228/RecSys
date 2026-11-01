# Данные

**Студент:** Леонтьев Константин Валерьевич

**Датасет:** Goodreads Book Graph (UCSD), жанровое подмножество *Comics & Graphic* — полки пользователей
(добавления, оценки 0–5, отметки о прочтении, даты) и метаданные книг (авторы, серии, издатель, формат, язык,
год, число страниц, пользовательские теги-полки, `work_id`).

* Страница датасета: https://mengtingwan.github.io/data/goodreads.html
* Файлы (публичные прямые ссылки):
  * https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_interactions_comics_graphic.json.gz
  * https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_books_comics_graphic.json.gz
* Статьи: M. Wan, J. McAuley. *Item Recommendation on Monotonic Behavior Chains*. RecSys 2018;
  M. Wan, R. Misra, N. Nakashole, J. McAuley. *Fine-Grained Spoiler Detection from Large-Scale Review Corpora*. ACL 2019.

Данные скачиваются автоматически функцией `models.data.download()` (первая ячейка раздела 1 ноутбука)
в папку `data/`, которая исключена из git.
