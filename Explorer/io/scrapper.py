from time import sleep
import igraph as ig
from collections import defaultdict
import requests
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
from tqdm import tqdm
import re

class TwoWayProgressBar:

    def __init__(self) -> None:
        self.ppb = None
        self.count = 0
        self.primary_size = 0

    def add_primary_pb(self):
        if self.ppb is not None:
            self.ppb.close()
        self.ppb = tqdm(total = self.primary_size)
        self.count = 0
        self.primary_size = 0

    def next_p(self):
        try:
            self.ppb.update(1)
            self.count += 1
            if self.count % 2000 == 0:
                return True
        except:
            print("failed")
        return False

    def finish_p(self):
        self.ppb.close()
    
class Scrapper:
    def __init__(self, base_url: str) -> None:
        self.graph_count = 0

        self.url = base_url
        try:
            self.graph = ig.load("./khan-academy.gml")
        except:
            self.graph = ig.Graph(directed = True)
        self.pb = TwoWayProgressBar()
    
    def add_vertex(self, parent: str, child: str) -> bool:
        # check if it already exist
        #   if not - create, and connect to parent
        #   if yes - just connect to parent if not yet connected
        # if both exists, and connected to that parent, return False, otherwise True

        try:
            self.graph.vs.find(name = child)
            v_exist = True
        except:
            v_exist = False

        if v_exist:
            e_exist = self.graph.are_connected(parent, child)
        else:
            e_exist = False
        
        if v_exist is False:
            self.graph.add_vertex(name = child)
        if e_exist is False:
            self.graph.add_edge(source = parent, target = child)

        if v_exist and e_exist:
            return False
        return True

    def add_vertices(self, parent: str, children: list[str]) -> list[bool]:
        return [self.add_vertex(parent, child) for child in children]
    
    def start(self, parent: str):
        # Starts the scrapper
        self.graph.add_vertex(name = parent)
        parent_crawler = self.crawl(parent)

        alive = True
        while alive:
            prompt = input("\nContinue? [y/n]: ")
            if prompt == "n":
                break
            
            self.pb.add_primary_pb()
            alive = next(parent_crawler)
            self.pb.finish_p()
        self.save_graph()
    
    def crawl(self, parent) -> bool:
        self.pb.primary_size += 1
        yield True

        if self.already_crawled(parent) is True:
            hrefs = [v["name"] for v in self.graph.vs.find(name = parent).successors()]
            crawlers = [self.crawl(href) for href in hrefs]
            alive_crawlers = [True for h in hrefs]
        else:
            raw_html = self.get_html(self.url + parent)
            hrefs = self.get_hrefs(raw_html)
            crawlers = [self.crawl(href) for href in hrefs]
            alive_crawlers = self.add_vertices(parent, hrefs)

        to_save = self.pb.next_p()
        if to_save is True:
            print("saving...")
            self.save_graph()

        while any(alive_crawlers):
            for i, c in enumerate(crawlers):
                if not alive_crawlers[i]:
                    continue
                try:
                    alive_crawlers[i] = next(c)
                except:
                    alive_crawlers[i] = False

            yield any(alive_crawlers)
    
    def already_crawled(self, vertex: str) -> bool:
        # https://python.igraph.org/en/stable/api/igraph.Vertex.html#outdegree
        try:
            if self.graph.vs.find(name = vertex).outdegree() > 0:
                return True
            return False
        except:
            return False

    def get_html(self, url) -> str:
        resp = requests.get(url)
        if resp.status_code == 200:
            return resp.text
        else:
            print(f"Failed accessing {url}. Not retrying.")

    def get_hrefs(self, html: str) -> list[str]:
        hrefs = set()
        soup = BeautifulSoup(html, 'html.parser')
        anchor_elements = soup.find_all('a')
        for element in anchor_elements:
            href = element.get('href')
            if len(href) == 0 or href[0] != "/":
                continue
            hrefs.add(href)
        return list(hrefs)

    def show_graph(self):
        fig, ax = plt.subplots()
        ig.plot(
            self.graph,
            target=ax,
            layout="sugiyama",
            vertex_size=15,
            vertex_color="grey",
            edge_color="#222",
            edge_width=1,
        )
        plt.show()
    
    def save_graph(self):
        for v in self.graph.vs:
            if v["id"] is None:
                v["id"] = 0

        self.graph.save(f"./khan-academy-{self.graph_count}.gml")
        self.graph_count += 1

def show_graph():
    g = ig.load("./khan-academy-2.gml")
    communities = g.community_edge_betweenness()
    communities = communities.as_clustering()

    num_communities = len(communities)
    palette1 = ig.RainbowPalette(n=num_communities)
    for i, community in enumerate(communities):
        g.vs[community]["color"] = i
        community_edges = g.es.select(_within=community)
        community_edges["color"] = i
    g.vs["name"] = ["\n\n" + label for label in g.vs["name"]]
    fig1, ax1 = plt.subplots()
    ig.plot(
        communities,
        target=ax1,
        mark_groups=True,
        palette=palette1,
        vertex_size=15,
        edge_width=0.5,
        vertex_label=[f"\n\n{int(i)}" for i in g.vs["id"]],
    )
    plt.show()

def count_profiles():
    g = ig.load("./khan-academy.gml")
    profile_counter = 0
    total_counter = 0
    for v in g.vs:
        total_counter += 1
        name: str = v["name"]
        if name.startswith("/profile/") is True:
            profile_counter += 1
    
    print(f"total_counter - {total_counter}")
    print(f"profile_counter - {profile_counter}")
    print(f"ratio - {profile_counter / total_counter}")
    print()

def first_level_uris():
    g = ig.load("./khan-academy.gml")
    uri_template = re.compile("\/[\w\-]*[\/\?]*")
    uris = defaultdict(int)
    total_count = 0

    for v in g.vs:
        total_count += 1
        name = v["name"]
        match = uri_template.match(name)
        if match is not None:
            uris[match[0]] += 1
        
    print("First level uris:")
    for k, v in sorted(uris.items(), key = lambda x: x[1]):
        print(f"\t'{k}': {v}, {v/total_count * 100}%")
    print()

class PopularityStack:
    def __init__(self, size) -> None:
        self.size = size
        self.stack = []
    
    def add(self, name, popularity):
        if len(self.stack) < self.size:
            self.stack.append((name, popularity))
            self.stack.sort(key = lambda x: x[1])
            return
        if self.stack[0][1] >= popularity:
            return
        self.stack.append((name, popularity))
        self.stack.sort(key = lambda x: x[1])
        if len(self.stack) <= self.size:
            return
        self.stack.pop(0)

def most_referenced():
    g = ig.load("./khan-academy.gml")
    popular_links = PopularityStack(100)
    for v in g.vs:
        name = v["name"]
        indegree = v.indegree()
        popular_links.add(name, indegree)
    
    print("Most referenced links:")
    for l in popular_links.stack:
        print(f"\t'{l[0]}': {l[1]}")

def most_referenced_first_level_uris():
    g = ig.load("./khan-academy.gml")
    uri_template = re.compile("\/[\w\-]*[\/\?]*")
    uris = defaultdict(int)
    popular_links = PopularityStack(30)
    for v in g.vs:
        name = v["name"]
        match = uri_template.match(name)
        if match is not None:
            indegree = v.indegree()
            uris[match[0]] += indegree
    
    for k, v in uris.items():
        popular_links.add(k, v)
    
    print("Most referenced first-level links:")
    for l in popular_links.stack:
        print(f"\t'{l[0]}': {l[1]}")

def generate_scanning_links():
    g = ig.load("./khan-academy.gml")
    uri_template = re.compile("(\/[\w\-]*)[\/\?]*")
    uris = {}
    uris["total"] = 0

    for v in g.vs:
        uris["total"] += 1
        name = v["name"]
        raw_match = uri_template.match(name)
        if raw_match is None:
            continue
        match = raw_match.group(1)
        if match not in uris:
            uris[match] = {}
            uris[match]["links"] = []
            uris[match]["count"] = 0
        uris[match]["links"].append("https://khanacademy.org" + name)
        uris[match]["count"] += 1
        
    return uris