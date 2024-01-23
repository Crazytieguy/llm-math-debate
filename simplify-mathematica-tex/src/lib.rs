use anyhow::anyhow;
use pyo3::{exceptions::PyValueError, prelude::*};
use serialize::serialize;
use simplify::simplify;
use winnow::Parser;

mod parse;
mod serialize;
mod simplify;

fn simplify_tex(tex: &str) -> anyhow::Result<String> {
    let parsed = parse::nodes
        .parse(tex)
        .map_err(|err| anyhow!("Error parsing TeX\n{}", err))?;
    let simplified = simplify(parsed)?;
    Ok(serialize(simplified))
}

/// Formats the sum of two numbers as string.
#[pyfunction]
#[pyo3(name = "simplify_tex")]
fn simplify_tex_py(py: Python<'_>, tex: &str) -> PyResult<String> {
    py.allow_threads(|| simplify_tex(tex).map_err(|err| PyValueError::new_err(err.to_string())))
}

/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
fn simplify_mathematica_tex(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(simplify_tex_py, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simplify_tex() {
        let input = r"\begin{array}{l}
    \begin{array}{l}
        \text{Simplify the following}: \\
        \sqrt{61}+\sqrt{125}           \\
    \end{array}
    \\
    \hline
    \begin{array}{l}
        \sqrt{125} \text{= }\sqrt{5^3} \text{= }5 \sqrt{5}: \\
        \fbox{$
                \begin{array}{ll}
                    \text{Answer:} &                               \\
                    \text{}        & \sqrt{61}+\fbox{$5 \sqrt{5}$} \\
                \end{array}
        $}                                                  \\
    \end{array}
    \\
\end{array}";
        let expected = r"Simplify the following:

$\sqrt{61}+\sqrt{125}$

\hrule

1. $\sqrt{125}\text{ = }\sqrt{5^3}\text{ = }5 \sqrt{5}$:

Answer: $\sqrt{61}+5 \sqrt{5}$";
        let actual = simplify_tex(input).unwrap();
        assert_eq!(actual, expected);
    }
}
