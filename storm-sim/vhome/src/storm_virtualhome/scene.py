"""Select reachable object/container pairs from a VirtualHome scene graph."""
from __future__ import annotations

import math
import random


def nodes_by_id(graph):
    return {node['id']: node for node in graph['nodes']}


def room_id_of(graph, node_id):
    """Resolve room identity without merging rooms that share a class name."""
    nodes = nodes_by_id(graph)
    parents = {}
    for edge in graph['edges']:
        if edge['relation_type'] == 'INSIDE':
            parents.setdefault(edge['from_id'], []).append(edge['to_id'])
    pending, seen = [node_id], set()
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        node = nodes.get(current, {})
        if str(node.get('category', '')).lower() == 'rooms':
            return node['id']
        pending.extend(parents.get(current, []))
    return None


def room_of(graph, node_id):
    room_id = room_id_of(graph, node_id)
    return nodes_by_id(graph)[room_id]['class_name'] if room_id is not None else None


def support_of(graph, node_id):
    nodes = nodes_by_id(graph)
    for edge in graph['edges']:
        if edge['from_id'] == node_id and edge['relation_type'] == 'ON':
            node = nodes[edge['to_id']]
            if 'SURFACES' in node.get('properties', []):
                return node
    return None


def position(node):
    return node.get('bounding_box', {}).get('center', node.get('obj_transform', {}).get('position', [0, 0, 0]))


def candidates(graph, room='kitchen', seed=0):
    containers = [n for n in graph['nodes'] if n['class_name'] in ('fridge', 'microwave', 'cabinet', 'kitchencabinet')
                  and 'CAN_OPEN' in n.get('properties', []) and 'CONTAINERS' in n.get('properties', [])
                  and room_of(graph, n['id']) == room]
    containers.sort(key=lambda n: (n['class_name'] != 'fridge', n['class_name'] != 'microwave', n['id']))
    objects = []
    for node in graph['nodes']:
        if 'GRABBABLE' not in node.get('properties', []) or room_of(graph, node['id']) != room:
            continue
        surface = support_of(graph, node['id'])
        if surface is None or 'CAN_OPEN' in surface.get('properties', []):
            continue
        size = node.get('bounding_box', {}).get('size', [0.1, 0.1, 0.1])
        if max(size) > 0.6:
            continue
        objects.append((node, surface))
    random.Random(seed).shuffle(objects)
    if containers:
        origin = position(containers[0])
        objects.sort(key=lambda pair: math.dist(position(pair[0]), origin))
    return containers, objects


def action(verb, *nodes, character=0):
    suffix = ' '.join(f"<{node['class_name']}> ({node['id']})" for node in nodes)
    return f'<char{character}> [{verb}] {suffix}'.strip()


def inside_closed(graph, target_id, container_id):
    nodes = nodes_by_id(graph)
    contained = any(e['from_id'] == target_id and e['to_id'] == container_id and e['relation_type'] == 'INSIDE'
                    for e in graph['edges'])
    return contained and 'CLOSED' in nodes[container_id].get('states', [])


def on_surface(graph, target_id, surface_id):
    return any(e['from_id'] == target_id and e['to_id'] == surface_id and e['relation_type'] == 'ON'
               for e in graph['edges'])


def held_by(graph, target_id, character_id=2):
    return any(e['from_id'] == character_id and e['to_id'] == target_id and
               e['relation_type'] in ('HOLDS_RH', 'HOLDS_LH') for e in graph['edges'])
