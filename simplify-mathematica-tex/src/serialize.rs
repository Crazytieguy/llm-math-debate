use crate::parse::{Array, Brackets, Curlies, Math, Node, Tag, TagContent, Text};

pub fn serialize(nodes: Vec<Node>) -> String {
    let mut buf = String::new();
    for node in nodes {
        serialize_node(node, &mut buf);
    }
    buf
}

fn serialize_node(node: Node, buf: &mut String) {
    match node {
        Node::Array(Array { align, elements }) => {
            buf.push_str(r"\begin{array}{");
            buf.push_str(align);
            buf.push_str("}\n");
            for element in elements {
                serialize_node(element, buf);
            }
            buf.push_str(r"\end{array}");
        }
        Node::Text(Text(elements)) => {
            buf.push_str(r"\text{");
            for element in elements {
                serialize_node(element, buf);
            }
            buf.push('}');
        }
        Node::Tag(Tag {
            tag,
            content,
            second_content,
        }) => {
            buf.push('\\');
            buf.push_str(tag);
            for content in [content, second_content] {
                match content {
                    Some(TagContent::Curlies(Curlies(nodes))) => {
                        buf.push('{');
                        for node in nodes {
                            serialize_node(node, buf);
                        }
                        buf.push('}');
                    }
                    Some(TagContent::Brackets(Brackets(nodes))) => {
                        buf.push('[');
                        for node in nodes {
                            serialize_node(node, buf);
                        }
                        buf.push(']');
                    }
                    None => (),
                }
            }
        }
        Node::Math(Math { content }) => {
            buf.push('$');
            for element in content {
                serialize_node(element, buf);
            }
            buf.push('$');
        }
        Node::Curlies(Curlies(nodes)) => {
            buf.push('{');
            for node in nodes {
                serialize_node(node, buf);
            }
            buf.push('}');
        }
        Node::Brackets(Brackets(nodes)) => {
            buf.push('[');
            for node in nodes {
                serialize_node(node, buf);
            }
            buf.push(']');
        }
        Node::LineBreak => buf.push_str(r"\\"),
        Node::Ampersand => buf.push('&'),
        Node::Raw(s) => buf.push_str(&s),
        Node::Escaped(c) => {
            buf.push('\\');
            buf.push(c);
        }
        Node::Space(s) => buf.push_str(s),
        Node::NewLine => buf.push('\n'),
    }
}
