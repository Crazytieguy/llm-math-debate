use std::io::{stdin, Read};

fn main() -> anyhow::Result<()> {
    let mut input = String::new();
    stdin().read_to_string(&mut input)?;
    let output = simplify_mathematica_tex::simplify_tex(&input)?;
    println!("{output}");
    Ok(())
}
