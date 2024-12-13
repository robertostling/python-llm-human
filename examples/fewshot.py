"""Demonstration of aspect-based sentiment analysis, with a twist.

We use few-shot examples to demonstrate how to avoid giving outright negative
answers.
"""

import plh

world = plh.PLH()


@world.llm()
def polite_assessment(text, person) -> str:
    """Obtain a polite assessment of a person given a text.

    Use `text` to figure out what the author thinks about the person
    specified in `person`.
    """
    pass


def training_data():
    fs = plh.FewShot(world)
    text = ('Former president Bashar al-Assad is a corrupt dictator and a '
            'murderer. Malala Yousafzai is the youngest recipient of the '
            'Nobel peace prize, and a world-famous advocate of women\'s '
            'rights.')
    fs.llm_call('polite_assessment', text, 'President al-Assad')
    fs.llm_return('polite_assessment', 'Not so positive.')
    fs.llm_call('polite_assessment', text, 'Malala')
    fs.llm_return('polite_assessment', 'Positive.')
    return fs.messages


print(polite_assessment(
    'Dr. Mengele was perhaps the most unethical "researcher" in history.',
    'Josef Mengele',
    _context=training_data()))
