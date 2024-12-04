//
// This script is used by pr.yml to determine if a domain must be tested based
// on the files modified in the pull request.
//

// Given a domain name and set of files, return true if the domain should be
// tested
function matchesPattern(domain, filePaths) {
  // filter files that end in .md
  filePaths = filePaths.filter(
    (filePath) =>
      !filePath.endsWith(".md") &&
      !filePath.startsWith("docs/") &&
      !filePath.startsWith("third-party-programs/"),
  );
  // These directories contain domain specific code
  const dirs = "(tests/unit_tests|examples|src|include/oneapi/math)";
  const domains = "(blas|lapack|rng|dft)";
  // matches changes to the domain of interest or non domain-specific code
  const re = new RegExp(`^(${dirs}/${domain}|(?!${dirs}/${domains}))`);
  const match = filePaths.some((filePath) => re.test(filePath));
  return match;
}

// Return the list of files modified in the pull request
async function prFiles(github, context) {
  let allFiles = [];
  let page = 0;
  let filesPerPage = 100; // GitHub's maximum per page

  while (true) {
    page++;
    const response = await github.rest.pulls.listFiles({
      owner: context.repo.owner,
      repo: context.repo.repo,
      pull_number: context.payload.pull_request.number,
      per_page: filesPerPage,
      page: page,
    });

    if (response.data.length === 0) {
      break; // Exit the loop if no more files are returned
    }

    allFiles = allFiles.concat(response.data.map((file) => file.filename));

    if (response.data.length < filesPerPage) {
      break; // Exit the loop if last page
    }
  }

  return allFiles;
}

// Called by pr.yml. See:
// https://github.com/actions/github-script/blob/main/README.md for more
// information on the github and context parameters
module.exports = async ({ github, context, domain }) => {
  if (!context.payload.pull_request) {
    console.log("Not a pull request. Testing all domains.");
    return true;
  }
  const files = await prFiles(github, context);
  const match = matchesPattern(domain, files);
  console.log("Domain: ", domain);
  console.log("PR files: ", files);
  console.log("Match: ", match);
  return match;
};

//
// Test the matchesPattern function
//
// Run this script with `node domain-check.js` It should exit with code 0 if
// all tests pass.
//
// If you need to change the set of files that are ignored, add a test pattern
// below with positive and negative examples. It is also possible to test by
// setting up a fork and then submitting pull requests that modify files, but
// it requires a lot of manual work.
//
test_patterns = [
  {
    domain: "blas",
    files: ["tests/unit_tests/blas/test_blas.cpp"],
    expected: true,
  },
  {
    domain: "rng",
    files: ["examples/rng/example_rng.cpp"],
    expected: true,
  },
  {
    domain: "lapack",
    files: ["include/oneapi/math/lapack/lapack.hpp"],
    expected: true,
  },
  {
    domain: "dft",
    files: ["src/dft/lapack.hpp"],
    expected: true,
  },
  {
    domain: "dft",
    files: ["src/dft/lapack.md"],
    expected: false,
  },
  {
    domain: "blas",
    files: ["tests/unit_tests/dft/test_blas.cpp"],
    expected: false,
  },
  {
    domain: "rng",
    files: ["examples/blas/example_rng.cpp"],
    expected: false,
  },
  {
    domain: "lapack",
    files: ["include/oneapi/math/rng/lapack.hpp"],
    expected: false,
  },
  {
    domain: "dft",
    files: ["src/lapack/lapack.hpp"],
    expected: false,
  },
  {
    domain: "dft",
    files: ["docs/dft/dft.rst"],
    expected: false,
  },
  {
    domain: "dft",
    files: ["third-party-programs/dft/dft.rst"],
    expected: false,
  },
];

function testPattern(test) {
  const result = matchesPattern(test.domain, test.files);
  if (result !== test.expected) {
    console.log("Fail:");
    console.log("  domain:", test.domain);
    console.log("  files:", test.files);
    console.log("  expected:", test.expected);
    console.log("  result:", result);
    process.exit(1);
  }
}

if (require.main === module) {
  // invoke test for each test pattern
  test_patterns.forEach(testPattern);
  console.log("All tests pass");
}
