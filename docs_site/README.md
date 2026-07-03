# DataFlex-RL Documentation

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

To contribute these pages upstream to DataFlex-Doc as a PR, the frontmatter + layout
already match its convention; adjust `permalink` prefixes and drop into its `docs/{en,zh}/notes/guide`.
