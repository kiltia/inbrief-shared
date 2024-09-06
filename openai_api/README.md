# OpenAI API

This module provides interaction with OpenAI API in our use-cases: `summarizing`, `editing`, `title generation` and use using on project building stage.

## Use-cases

All functions that provide interaction with OpenAI API are located in `core.py` file. Prompts for functions are located in `prompts.py`.

### Summarizing

This use-case is consist in summarizing text. To solve this problem, use the `summarize()` and `asummarize()`(asynchronous version of the function `summarize()`) functions.

### Editing

This use-case is consist in editing the text by correcting grammatical and lexical errors and paraphrasing the text in a given style. To solve this problem, use the `edit()` and `aedit()`(asynchronous version of the function `edit()`) functions.

### Title generation

This use-case is consist in generating a title for a given text. To solve this problem, use the `get_title()` and `aget_title()`(asynchronous version of the function `get_title()`) functions.