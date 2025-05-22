# Backend Maintainer Roles

### Revision


|Date       |Revision | Comments                                                                 |
|-----------|---------|--------------------------------------------------------------------------|
|  20250509 |  1.0    | Initial version                                                          |
|  20250522 |  1.1    | Include CODEOWNERS file in project changes                               |

## Motivation

Currently the maintainer role is associated either with a particular domain,
like BLAS or DFT, or the overall Architecture maintainer role.
However, as more and more backends are supported in oneMath, it makes sense
to add additional maintainer roles targeting particular backends.
Various improvements and fixes to the oneMath implementation are specific
to a particular backend, which requires expertise for that architecture,
which is different than the domain expertise required for a particular domain.
Supporting the maintainer role for different backends will allow for broader
ecosystem support by those who are most familiar with specific backends.
As an example, the pull request enabling CI testing for ArmPL backend
[#668](https://github.com/uxlfoundation/oneMath/pull/668) was reviewed
by the person who contributed ArmPL backend support for BLAS, LAPACK, and RNG
domains. If the backend maintainer role existed, then this review could also
have been a formal approval of the pull request.

**We aim to make a decision on the backend maintainer role by May 31, 2025.**

## Outline

1. [Introduction](#introduction)
2. [Proposal](#proposal)
3. [Changes in the Project](#changes-in-the-project)
4. [Open questions](#open-questions)

## Introduction

The maintainer role is currently associated either with a particular domain
or the overall Architecture maintainer role. As more backends are supported
in oneMath, it makes sense to add additional maintainer roles targeting
particular backends.

## Proposal

We propose to introduce the following backend-specific roles:
* @uxlfoundation/onemath-cpu-aarch64
* @uxlfoundation/onemath-cpu-x64
* @uxlfoundation/onemath-gpu-amd
* @uxlfoundation/onemath-gpu-intel
* @uxlfoundation/onemath-gpu-nvidia

As additional backends are enabled, additional roles can be added.
We will use the same process as currently done for Domain Maintainers
for the new Backend Maintainers.

### Other Considered Approaches

If we do not introduce these backend-specific roles, then contributors who
want to take on a maintainer role will need to become proficient in at least
one specific domain of the project.

## Changes in the Project

The changes will be in the Github roles and CODEOWNERS as well as changes in
the MAINTAINERS.md and CONTRIBUTING.md files to describe these new roles.

## Open questions

* Should backend maintainer roles be created for backends not currently
  supported in oneMath?
  * We propose to initially create backend maintainer roles only for backends
    currently supported in oneMath.
