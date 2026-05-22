# Contribute to TMRL

Thank you for contributing to the `tmrl` library.

We do manually review each and every line in your PRs before merging anything.
Therefore, we ask that you stick as much as possible to the following guidelines in order to minimize our workload.

## Important guidelines for Pull Requests (PRs):

### :one: One self-contained concern per PR:
Your PRs will most likely be rejected or manually re-implemented if they tackle several concerns at once.

For instance, let us imagine that you want to refactor part of the codebase and add support for two new RL algorithms.
Then you should create:
- (a) One PR that only implements refactoring,
- (b) One PR that only implements support for algorithm 1, based on PR (a),
- (c) One PR that only implements support for algorithm 2, based on PR (b).

### :two: Keep each PR line count minimal:
Roughly speaking, a PR should change less that 10 lines if fixing a small concern and less than 250 lines if introducing a major feature. Of course this may vary depending on what the PR is implementing, but in most cases a PR of more than 250 lines should probably be broken into several smaller PRs by following guideline (1).

### :three: No AI-generated code:
We insist that **we do manually and thoroughly review each single line of code** that your submit in your PRs.
This takes infinitely more time than asking an LLM agent to write a feature and trusting that it is correct, so please refrain from doing this entirely.

Whenever asking an LLM to draft a feature, please re-write each line of generated code yourself, making the code your own and making sure that you **fully** understand whatever you are submitting.

You can (and should) however ask LLMs to help debug your code before submitting a PR.

### :four: Do use AI to debug your code.
See guideline (3).

### :five: No automated re-formatting:
Do not submit any line change that doesn't do anything functionally useful, as typically generated automatically by formatting tools like Ruff.
PRs that include such changes will most likely be rejected.

If you believe that the codebase would benefit from changes in formatting, please discuss this in the [discussion section](https://github.com/trackmania-rl/tmrl/discussions).

### :six: Additional guidelines for comments, documentation and code formatting:
Adding/improving documentation and comments in the existing code is welcome, but please make sure that your PR follows guildelines (1) and (3).

Of course, when your PR is adding a feature or modifiying existing code, it should include the corresponding docstrings and comments.

**Docstrings format**: Please follow the [Google style format](https://gist.github.com/redlotus/3bc387c2591e3e908c9b63b97b11d24e) for documentation.

**No PEP8 79-character line cuts in code instructions:**
The `tmrl` codebase **does not** follow the 79-character limit defined in PEP8, and your code should not actively attempt to stick to this rule when contributing to this repository.
Instead, please try to stick to 1 instruction per line when reasonable, and cut lines only for optimal readability.
In general, do not use `\`.
For instance:

```python
# OK:
self.array_with_a_long_name = np.array([variable_1 + variable_2, (variable_3 * variable_4) ** variable_5])

# OK:
self.array_with_a_long_name = np.array(
    [
        variable_1 + variable_2,
        (variable_3 * variable_4) ** variable_5
    ]
)

# NOT OK:
self.array_with_a_long_name = np.array([variable_1 + variable_2, (variable_3 \
    * variable_4) ** variable_5])
```

**PEP8 for everything else:**
```python
# OK:
self.array_with_a_long_name = np.array([variable_1 + variable_2, (variable_3 * variable_4) ** variable_5])

# NOT OK:
self.arrayWithWLongName = np.array([variable1+variable2,(variable3*variable4)**variable5])
```

The [Google Python style guide](https://google.github.io/styleguide/pyguide.html) is a good resource for `tmrl` code formatting, just ignore the 80-character limit.