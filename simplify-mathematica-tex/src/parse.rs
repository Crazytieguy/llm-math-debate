use std::borrow::Cow;

use winnow::ascii::{alpha1, multispace0, space1};
use winnow::combinator::{alt, delimited, opt, preceded, repeat};
use winnow::token::{one_of, take_while};
use winnow::{seq, Parser};

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Node<'a> {
    Array(Array<'a>),
    Text(Text<'a>),
    Tag(Tag<'a>),
    Math(Math<'a>),
    Curlies(Curlies<'a>),
    Brackets(Brackets<'a>),
    LineBreak,
    Ampersand,
    Raw(Cow<'a, str>),
    Space(&'a str),
    NewLine,
    Escaped(char),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Text<'a>(pub(crate) Vec<Node<'a>>);

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Curlies<'a>(pub(crate) Vec<Node<'a>>);

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Brackets<'a>(pub(crate) Vec<Node<'a>>);

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Array<'a> {
    pub(crate) align: &'a str,
    pub(crate) elements: Vec<Node<'a>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum TagContent<'a> {
    Curlies(Curlies<'a>),
    Brackets(Brackets<'a>),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Tag<'a> {
    pub(crate) tag: &'a str,
    pub(crate) content: Option<TagContent<'a>>,
    pub(crate) second_content: Option<TagContent<'a>>, // for ie \frac{...}{...}
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Math<'a> {
    pub(crate) content: Vec<Node<'a>>,
}

pub(crate) fn nodes<'a>(input: &mut &'a str) -> winnow::PResult<Vec<Node<'a>>> {
    repeat(0.., node).parse_next(input)
}

fn node<'a>(input: &mut &'a str) -> winnow::PResult<Node<'a>> {
    alt((
        array,
        text,
        tag,
        math,
        curlies.map(Node::Curlies),
        brackets.map(Node::Brackets),
        r"\\".value(Node::LineBreak),
        '&'.value(Node::Ampersand),
        raw,
        space1.map(Node::Space),
        '\n'.value(Node::NewLine),
        escaped,
    ))
    .parse_next(input)
}

fn text<'a>(input: &mut &'a str) -> winnow::PResult<Node<'a>> {
    delimited(r"\text{", nodes, '}')
        .map(Text)
        .map(Node::Text)
        .parse_next(input)
}

fn array<'a>(input: &mut &'a str) -> winnow::PResult<Node<'a>> {
    seq! {Array {
       _: "\\begin{array}",
       align: delimited('{', take_while(1.., |c| c != '}'), '}'),
       _: multispace0,
       elements: nodes,
       _: multispace0,
       _: "\\end{array}",
    }}
    .map(Node::Array)
    .parse_next(input)
}

fn tag<'a>(input: &mut &'a str) -> winnow::PResult<Node<'a>> {
    seq! {Tag {
        tag: preceded(r"\", alpha1.verify(|name: &str| name != "begin" && name != "end")),
        content: opt(alt((
            curlies.map(TagContent::Curlies),
            brackets.map(TagContent::Brackets),
        ))),
        second_content: opt(alt((
            curlies.map(TagContent::Curlies),
            brackets.map(TagContent::Brackets),
        ))),
    }}
    .map(Node::Tag)
    .parse_next(input)
}

fn math<'a>(input: &mut &'a str) -> winnow::PResult<Node<'a>> {
    seq! {Math {
        content: delimited('$', nodes, '$'),
    }}
    .map(Node::Math)
    .parse_next(input)
}

fn curlies<'a>(input: &mut &'a str) -> winnow::PResult<Curlies<'a>> {
    delimited('{', nodes, '}').map(Curlies).parse_next(input)
}

fn brackets<'a>(input: &mut &'a str) -> winnow::PResult<Brackets<'a>> {
    delimited('[', nodes, ']').map(Brackets).parse_next(input)
}

fn raw<'a>(input: &mut &'a str) -> winnow::PResult<Node<'a>> {
    take_while(1.., |c| {
        !['\\', '$', '{', '}', '[', ']', '&', ' ', '\n'].contains(&c)
    })
    .map(Cow::Borrowed)
    .map(Node::Raw)
    .parse_next(input)
}

fn escaped<'a>(input: &mut &'a str) -> winnow::PResult<Node<'a>> {
    preceded('\\', one_of(['{', '}', ' ', ',', '|', '_', '!']))
        .map(Node::Escaped)
        .parse_next(input)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parsing() -> anyhow::Result<()> {
        let input = r"\begin{array}{l}
    Hello World
\end{array}";
        let expected = vec![Node::Array(Array {
            align: "l",
            elements: vec![
                Node::Raw(Cow::Borrowed("Hello")),
                Node::Space(" "),
                Node::Raw(Cow::Borrowed("World")),
                Node::NewLine,
            ],
        })];
        let actual = nodes
            .parse(input)
            .map_err(|err| anyhow::anyhow!("Failed parsing: {err}"))?;
        assert_eq!(actual, expected);
        let input = r"Are the following numbers relatively prime (coprime)? $\{185,416\}$.";
        let expected = vec![
            Node::Raw(Cow::Borrowed("Are")),
            Node::Space(" "),
            Node::Raw(Cow::Borrowed("the")),
            Node::Space(" "),
            Node::Raw(Cow::Borrowed("following")),
            Node::Space(" "),
            Node::Raw(Cow::Borrowed("numbers")),
            Node::Space(" "),
            Node::Raw(Cow::Borrowed("relatively")),
            Node::Space(" "),
            Node::Raw(Cow::Borrowed("prime")),
            Node::Space(" "),
            Node::Raw(Cow::Borrowed("(coprime)?")),
            Node::Space(" "),
            Node::Math(Math {
                content: vec![
                    Node::Escaped('{'),
                    Node::Raw(Cow::Borrowed("185,416")),
                    Node::Escaped('}'),
                ],
            }),
            Node::Raw(Cow::Borrowed(".")),
        ];
        let actual = nodes
            .parse(input)
            .map_err(|err| anyhow::anyhow!("Failed parsing: {err}"))?;
        assert_eq!(actual, expected);
        Ok(())
    }
}
