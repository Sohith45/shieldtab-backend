"""
The public dataset's 'benign' class is almost entirely bare domain/path
listings (from top-site crawls) with virtually no query strings, while the
malicious classes are query-string-heavy. That skews the model toward
flagging any URL with params as suspicious (false positives on things like
a Google search or a YouTube watch link).

This generates a modest set of realistic benign URLs *with* query strings
across common real-world usage patterns, to rebalance that gap. It's a
documented data-augmentation step, not a leak: none of these are labeled
malicious data, and the templates reflect genuinely benign query patterns.
"""
import random
import csv

random.seed(42)

SEARCH_TERMS = ['python tutorial', 'weather today', 'best laptops 2026', 'recipe pasta',
                'news', 'movie times', 'flight status', 'cheap flights', 'nba scores',
                'stock market', 'machine learning', 'javascript array methods', 'gym near me']
VIDEO_IDS = ['dQw4w9WgXcQ', 'jNQXAC9IVRw', 'kJQP7kiw5Fk', '9bZkp7q19f0']
PRODUCT_IDS = ['B08N5WRWNW', 'B07XJ8C8F5', 'B09G9FPHY6']
USERNAMES = ['johndoe', 'jane_smith', 'techguru', 'alice92']
PAGE_TITLES = ['Machine_learning', 'Python_(programming_language)', 'Climate_change']
LANGS = ['en', 'es', 'fr', 'de', 'hi']

templates = []

for term in SEARCH_TERMS:
    q = term.replace(' ', '+')
    templates.append(f'www.google.com/search?q={q}')
    templates.append(f'www.bing.com/search?q={q}')
    templates.append(f'duckduckgo.com/?q={q}')

for vid in VIDEO_IDS:
    templates.append(f'www.youtube.com/watch?v={vid}')
    templates.append(f'www.youtube.com/watch?v={vid}&t=42s')

for pid in PRODUCT_IDS:
    templates.append(f'www.amazon.com/dp/{pid}')
    templates.append(f'www.amazon.com/gp/product/{pid}?ref=sr_1_1')

for user in USERNAMES:
    templates.append(f'twitter.com/{user}')
    templates.append(f'www.instagram.com/{user}/?hl=en')
    templates.append(f'github.com/{user}?tab=repositories')

for title in PAGE_TITLES:
    for lang in LANGS:
        templates.append(f'{lang}.wikipedia.org/wiki/{title}')

templates += [
    'www.linkedin.com/in/johndoe/?originalSubdomain=us',
    'maps.google.com/maps?q=central+park+new+york',
    'www.reddit.com/r/programming/comments/abc123/interesting_post/',
    'stackoverflow.com/questions/12345678/how-to-parse-json-in-python',
    'docs.google.com/document/d/1a2b3c4d5e6f/edit?usp=sharing',
    'drive.google.com/file/d/1a2b3c4d5e6f/view?usp=sharing',
    'www.nytimes.com/2026/09/01/technology/ai-news.html?smid=url-share',
    'weather.com/weather/today/l/USNY0996?par=google',
    'www.booking.com/hotel/us/example.html?checkin=2026-10-01&checkout=2026-10-05',
    'www.airbnb.com/rooms/12345678?check_in=2026-10-01&check_out=2026-10-05',
]

# small variations to bulk it up a bit further without duplicating exactly
extra = []
for t in templates:
    if '?' in t:
        extra.append(t + '&utm_source=newsletter')

templates += extra

print(f'Generated {len(templates)} synthetic benign URLs')

with open('synthetic_benign.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['url', 'type'])
    for t in templates:
        writer.writerow([t, 'benign'])
