# DataFlex-RL Documentation

The maintained documentation site is published at
<https://haolpku.github.io/DataFlex-RL-Doc>. This directory is a lightweight
mirror for readers browsing the code repository; use the separate documentation
repository for site changes and the full bilingual guide.

VuePress + [plume theme](https://theme-plume.vuejs.press/) site, bilingual (en/zh),
mirroring the DataFlex-Doc structure. Content lives in `docs/en` and `docs/zh`;
sidebar/navbar in `docs/.vuepress`.

```bash
cd docs_site
npm install
npm run docs:dev     # local preview at http://localhost:8080
npm run docs:build   # static build -> docs/.vuepress/dist
```

Structure (mirrors https://github.com/OpenDCAI/DataFlex-Doc):
- `basicinfo/` — intro, framework design, install
- `reweighter/` `selector/` `mixer/` — each: quickstart, tutorial (how to add), algorithm pages

The frontmatter and layout match the standalone documentation repository, so pages
can be copied there when the site is updated.
