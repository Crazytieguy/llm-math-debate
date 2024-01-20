use std::borrow::Cow;

use anyhow::{anyhow, bail};
use itertools::Itertools;

use crate::parse::*;

pub fn simplify(parsed_tex: Vec<Node>) -> anyhow::Result<Vec<Node>> {
    let mut flattened = flatten_and_number_solution_steps(parsed_tex)?;
    unwrap_answer_box(&mut flattened)?;
    let fixed = fix_text_math_contexts(flattened);
    let unwraped = unwrap_fboxes_and_text(fixed);
    let cleaned = clean_whitespace(unwraped);
    Ok(cleaned)
}

fn unwrap_fboxes_and_text(nodes: Vec<Node>) -> Vec<Node> {
    nodes
        .into_iter()
        .flat_map(|elem| match elem {
            Node::Tag(Tag {
                tag: "fbox",
                content: Some(TagContent::Curlies(Curlies(fbox_content))),
                second_content: None,
            }) => {
                let Ok(Node::Math(Math { content })) = fbox_content.into_iter().exactly_one()
                else {
                    panic!("Expected fbox child to be Math");
                };
                unwrap_fboxes_and_text(content)
            }
            Node::Array(Array { align, elements }) => vec![Node::Array(Array {
                align,
                elements: unwrap_fboxes_and_text(elements),
            })],
            Node::Tag(Tag {
                tag,
                content,
                second_content,
            }) => {
                let content = content.map(|content| match content {
                    TagContent::Curlies(Curlies(nodes)) => {
                        TagContent::Curlies(Curlies(unwrap_fboxes_and_text(nodes)))
                    }
                    TagContent::Brackets(Brackets(nodes)) => {
                        TagContent::Brackets(Brackets(unwrap_fboxes_and_text(nodes)))
                    }
                });
                let second_content = second_content.map(|second_content| match second_content {
                    TagContent::Curlies(Curlies(nodes)) => {
                        TagContent::Curlies(Curlies(unwrap_fboxes_and_text(nodes)))
                    }
                    TagContent::Brackets(Brackets(nodes)) => {
                        TagContent::Brackets(Brackets(unwrap_fboxes_and_text(nodes)))
                    }
                });
                vec![Node::Tag(Tag {
                    tag,
                    content,
                    second_content,
                })]
            }
            Node::Math(Math { content }) => vec![Node::Math(Math {
                content: unwrap_fboxes_and_text(content),
            })],
            Node::Curlies(Curlies(content)) => {
                vec![Node::Curlies(Curlies(unwrap_fboxes_and_text(content)))]
            }
            Node::Brackets(Brackets(content)) => {
                vec![Node::Brackets(Brackets(unwrap_fboxes_and_text(content)))]
            }
            Node::Text(Text(s)) => {
                if !s.contains(char::is_alphabetic) {
                    vec![Node::Raw(s)]
                } else {
                    vec![Node::Text(Text(s))]
                }
            }
            _ => vec![elem],
        })
        .group_by(|elem| matches!(elem, Node::Space(_) | Node::Text(_)))
        .into_iter()
        .flat_map(|(is_contiguous_text_whitespace, elems)| {
            let elems = elems.collect_vec();
            if !is_contiguous_text_whitespace {
                return elems;
            }
            if elems.iter().all(|elem| matches!(elem, Node::Space(_))) {
                return elems;
            }
            vec![Node::Text(Text(Cow::Owned(
                elems
                    .into_iter()
                    .map(|elem| match elem {
                        Node::Space(s) => Cow::Borrowed(s),
                        Node::Text(Text(s)) => s,
                        _ => unreachable!(),
                    })
                    .collect::<String>(),
            )))]
        })
        .collect()
}

fn clean_whitespace(nodes: Vec<Node>) -> Vec<Node> {
    nodes
        .into_iter()
        .group_by(|elem| matches!(elem, Node::Space(_) | Node::NewLine))
        .into_iter()
        .flat_map(|(is_whitespace, mut group)| {
            if !is_whitespace {
                return group.collect_vec();
            }
            if group.contains(&Node::NewLine) {
                vec![Node::NewLine]
            } else {
                vec![Node::Space(" ")]
            }
        })
        .collect()
}

fn fix_text_math_contexts(nodes: Vec<Node>) -> Vec<Node> {
    nodes
        .into_iter()
        .group_by(|node| matches!(node, Node::LineBreak))
        .into_iter()
        .flat_map(|(is_line_break, group)| {
            if is_line_break {
                return vec![]; // discard line breaks for now
            }
            let mut elems = group.collect_vec();
            let mut after_math_part = vec![];
            let mut last_math_elem = None;
            while let Some(elem) = elems.pop() {
                match elem {
                    Node::Text(Text(s)) => {
                        after_math_part.insert(0, Node::Raw(s));
                    }
                    Node::Space(s) => after_math_part.insert(0, Node::Space(s)),
                    Node::NewLine => after_math_part.insert(0, Node::NewLine),
                    Node::Tag(Tag {
                        tag: "hline",
                        content: None,
                        second_content: None,
                    }) => {
                        after_math_part.insert(
                            0,
                            Node::Tag(Tag {
                                tag: "hrule",
                                content: None,
                                second_content: None,
                            }),
                        );
                    }
                    _ => {
                        last_math_elem = Some(elem);
                        break;
                    }
                }
            }
            let Some(last_math_elem) = last_math_elem else {
                return after_math_part;
            };
            let mut before_math_part = vec![];
            let mut math_content = vec![];
            for elem in elems {
                if !math_content.is_empty() {
                    math_content.push(elem);
                    continue;
                }
                match elem {
                    Node::Text(Text(s)) => {
                        before_math_part.push(Node::Raw(s));
                    }
                    Node::Space(s) => before_math_part.push(Node::Space(s)),
                    Node::NewLine => before_math_part.push(Node::NewLine),
                    Node::Tag(Tag {
                        tag: "hline",
                        content: None,
                        second_content: None,
                    }) => {
                        before_math_part.push(Node::Tag(Tag {
                            tag: "hrule",
                            content: None,
                            second_content: None,
                        }));
                    }
                    _ => {
                        math_content.push(elem);
                    }
                }
            }
            math_content.push(last_math_elem);
            let math_part = Node::Math(Math {
                content: math_content,
            });
            before_math_part
                .into_iter()
                .chain([math_part])
                .chain(after_math_part)
                .collect()
        })
        .collect()
}

fn unwrap_answer_box(flattened: &mut Vec<Node>) -> anyhow::Result<()> {
    let mut fbox_content = loop {
        if flattened.is_empty() {
            bail!("No fbox found");
        }
        if let Some(Node::Tag(Tag {
            tag: "fbox",
            content: Some(TagContent::Curlies(Curlies(fbox_content))),
            second_content: None,
        })) = flattened.pop()
        {
            break fbox_content;
        }
    };
    let Some(Node::Math(Math { content })) = fbox_content.pop() else {
        bail!("Expected fbox child to be Math");
    };
    let answer_array = content
        .into_iter()
        .find_map(|node| match node {
            Node::Array(Array {
                align: "ll",
                elements,
            }) => Some(elements),
            _ => None,
        })
        .ok_or_else(|| anyhow!("No answer array found"))?;
    let second_ampersand_position = answer_array
        .iter()
        .positions(|node| *node == Node::Ampersand)
        .nth(1)
        .ok_or_else(|| anyhow!("No second ampersand found"))?;
    flattened.push(Node::Text(Text(Cow::Borrowed("Answer:"))));
    flattened.extend(answer_array.into_iter().skip(second_ampersand_position + 1));
    Ok(())
}

fn flatten_and_number_solution_steps(parsed_tex: Vec<Node>) -> anyhow::Result<Vec<Node>> {
    let elements = parsed_tex
        .into_iter()
        .find_map(|node| match node {
            Node::Array(Array {
                align: "l",
                elements,
            }) => Some(elements),
            _ => None,
        })
        .ok_or_else(|| anyhow!("No top level answer array found"))?;
    let mut simplified = vec![];
    let mut arrays_found = 0;
    for element in elements {
        match element {
            Node::Array(Array {
                align: "l",
                elements: children,
            }) => {
                if arrays_found > 0 {
                    simplified.push(Node::Text(Text(Cow::Owned(format!("{arrays_found}. ")))));
                }
                simplified.extend(children);
                arrays_found += 1;
            }
            el => simplified.push(el),
        }
    }
    Ok(simplified)
}
