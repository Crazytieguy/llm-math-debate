use anyhow::anyhow;
use serialize::serialize;
use simplify::simplify;
use winnow::Parser;

mod parse;
mod serialize;
mod simplify;

pub fn simplify_tex(tex: &str) -> anyhow::Result<String> {
    let parsed = parse::nodes
        .parse(tex)
        .map_err(|err| anyhow!("Error parsing TeX: {}", err))?;
    let simplified = simplify(parsed)?;
    Ok(serialize(simplified))
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
