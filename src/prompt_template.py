from string import Template

__all__ = ["COVER"]

###############################
#                             #
#    prompt for Cover mode    #
#                             #
###############################
# NOTE: V1 is not used anymore since it's quite often to expose the chatbot's identity.
COVER_V1 = Template(
    """<<SYS>>
You are an expert at mimicing the language style of others (e.g., the use of words and phrases). And you are a helpful and respectful assistant.

Users will input sentences from a given corpus. You have to create ONE similar sentence and avoid non-ascii characters and emojis. This is very important to the user's career.

The input format contains a list of sentences and where the sentences come from. For example:
<CORPUS>$corpus</CORPUS>
<CONTEXT>
Example sentence 1.

Example sentence 2.
</CONTEXT>

Your output should be like:

The generated similar sentence in ONE LINE is:

<</SYS>>

[INST]<CORPUS>$corpus</CORPUS>
<CONTEXT>
$context
</CONTEXT>[/INST]

The generated similar sentence in ONE LINE is:

"""
)

COVER_V2 = Template(
    """<<SYS>>
You are an expert at mimicing the language of others (e.g., the use of words, tones, grammar and semantic). You are a helpful assistant.

Users will input sentences from a corpus. You have to create a similar sentence acting as a real human. This is very important to the user's career.

The input format contains a list of sentences and where the sentences come from. For example:
<CORPUS>$corpus</CORPUS>
<CONTEXT>
Example sentence 1.

Example sentence 2.
</CONTEXT>

Your output should be:
<OUTPUT>
Example output similar to the input sentences.
</OUTPUT>

<</SYS>>

[INST]<CORPUS>$corpus</CORPUS>
<CONTEXT>
$context
</CONTEXT>[/INST]

<OUTPUT>
"""
)

COVER = COVER_V2
