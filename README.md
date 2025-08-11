# News Links Aggregator (Free, RSS-based)

Собирает **только ссылки** на свежие тематические материалы из RSS-лент. Без платных API.

## Как работает
- Берёт список RSS из `sources.txt`.
- Фильтрует по ключевым словам из `keywords.txt` (если файл пуст — берёт всё).
- Исключает по `stopwords.txt` (опционально).
- Сортирует по времени и сохраняет **чистый список ссылок** в `links.txt` (одна ссылка в строку, максимум 200).

## Быстрый старт (локально)
1) Установите Python 3.10+.
2) `pip install -r requirements.txt`
3) Отредактируйте `sources.txt` и (по желанию) `keywords.txt`.
4) Запустите:  `python aggregator_links.py`
5) Готово: смотрите `links.txt`.

## GitHub Actions (автозапуск)
Workflow в `.github/workflows/build.yml` обновляет `links.txt` каждые 30 минут.
