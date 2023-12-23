CLASSIFY_TASK = "Представь, что ты классификатор. Я буду давай тебе текст, а ты отвечай мне названием класса к которой относится этот текст. В ответе выводи только его, без каких-либо дополнительных пояснений, извинений или иных слов."


SUMMARIZE_TASK = "Тебе необходимо суммаризовать новостной сюжет в новый текст, сохранив только САМЫЕ основные, САМЫЕ важные и САМЫЕ яркие для читателя тезисы, которые описывают сюжет. Суммаризация должна быть значительно меньше оригинального текста, при этом она также должна быть логически и грамматически связной, чтобы результат должно быть приятно и легко читать. Отвечай только суммаризацией, не добавляй никаких дополнительных комментариев и пояснений."


TITLE_TASK = "Представь, что ты генератор заголовков для телеграм постов. Тебе необходимо придумать заголовок к заданному тексту, который отражает его главный смысл. Отвечай только самим заголовком без дополнительных комментариев и пояснений."


EDITOR_TASK = "Представь, что ты редактор телеграм постов. Придерживайся следующих правил: 1. Удаляй повторяющиеся сообщения \n 2. Исправляй орфографические и грамматические ошибки. \n 3. Обязательно разделяй текст на осмысленные абзацы, чтобы улучшить читаемость. \n 4. Не добавляй никакой лишней информации, комментариев, пояснений и других подобных вещей от себя."


def get_summary_context(text, max_words):
    return [
        {"role": "system", "content": SUMMARIZE_TASK},
        {"role": "user", "content": text},
        {
            "role": "system",
            "content": f"Необходимо использовать строго не более {max_words} слов.",
        },
    ]


def get_title_context(text, max_words):
    return [
        {"role": "system", "content": TITLE_TASK},
        {"role": "user", "content": text},
        {
            "role": "system",
            "content": "Необходимо использовать строго не более {max_words} слов.",
        },
    ]


def get_editor_context(text, max_words, style):
    return [
        {"role": "system", "content": EDITOR_TASK},
        {"role": "user", "content": text},
        {
            "role": "system",
            "content": f"Дополнительные условия: необходимо использовать строго не более {max_words} слов. Придерживайся следующего стиля текста: {style}",
        },
    ]
