# Use of AI tools

This project is developed with the help of AI assistants. This note explains
how we use them and where responsibility lies. The same note applies to all
projects maintained by Tiago Tresoldi, unless stated otherwise.

## Design is ours

The research questions, the methods, and the design of the software (its
scope, architecture, interfaces, algorithms and defaults) are decided by the
researcher, acting as the software engineer of the project. AI tools do not
decide what the software does, which methods it implements, or how its results
should be interpreted.

## What we delegate

We use AI assistants mostly for routine engineering work, such as:

- continuous integration, packaging and release pipelines;
- tooling configuration, formatting and dependency updates;
- tests, including additional test cases for behaviour we specify;
- documentation structure, examples and consistency checks;
- code review of changes, as a second reader.

AI assistants may also draft implementations of designs we specify. Drafts are
reviewed, tested and often rewritten before they are merged; nothing enters the
repository without a person reviewing it.

## Writing

English is not our first language. We use AI assistants to review text we
write (documentation, README files, release notes and similar), for clarity,
grammar and style. The content, the claims and the choice of what to say are
ours.

## Data

AI tools are never used to create data. At most, they help us locate datasets
and obtain them from their sources, whose provenance and licences we record.

## Responsibility

We are responsible for everything in this repository, whether or not an AI
tool helped produce it: the code, the tests, the documentation, the reported
results and the references. Results are checked by tests and, where possible,
against published values or reference implementations, and every reference is
checked against its source. Errors are ours; please report them in the issue
tracker.

## Tools

We mainly use Anthropic's Claude, through Claude Code, and occasionally other
assistants. The specific models change over time.

AI tools are not authors of this software. To cite it, use the citation
information in the repository.

*Last reviewed: November 2026.*
