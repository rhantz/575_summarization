from collections import defaultdict

class ThemeGraph:
    
    def __init__(self):
        self.theme_graph = defaultdict(lambda: defaultdict(float))
    
    def add_arc_weight(self, from_theme, to_theme, weight):
        self.theme_graph[from_theme][to_theme] += weight
    
    def theme_order(self):
        
        theme_order = []
        # get all themes
        all_themes = set(self.theme_graph.keys())
        for from_theme, to_theme_and_weight in self.theme_graph.items():
            all_themes.update(set(to_theme_and_weight.keys()))
        
        # add arc between all themes (by adding a 0 weight)
        for from_theme in all_themes:
            for to_theme in all_themes:
                self.theme_graph[from_theme][to_theme] += 0      
        
        iterations = len(all_themes)
        for i in range(iterations):
            
            theme_weight_dict = defaultdict(float)
            
            # calculate the theme weight
            for from_theme, to_theme_and_weight in self.theme_graph.items():
                for to_theme, weight in to_theme_and_weight.items():
                    theme_weight_dict[from_theme] += weight
                    theme_weight_dict[to_theme] -= weight
            
            # get the themes with highest weight and add to theme order list
            highest_weighted_themes = [k for k, v in theme_weight_dict.items() if v == max(theme_weight_dict.values())]
            theme_order = theme_order + highest_weighted_themes
            
            # remove the added themes from theme graph
            for theme in highest_weighted_themes:
                del self.theme_graph[theme]
                
            for from_theme, to_theme_and_weight in self.theme_graph.items():
                for theme in highest_weighted_themes:
                    if theme in to_theme_and_weight.keys():
                        del to_theme_and_weight[theme]
                  
        return theme_order

if __name__ == "__main__":
    
    # initialize graph
    graph = ThemeGraph()

    # add arc weights
    graph.add_arc_weight("t1", "t2", 2)
    graph.add_arc_weight("t1", "t3", 2)
    graph.add_arc_weight("t2", "t3", 2)
    graph.add_arc_weight("t2", "t4", 1)
    graph.add_arc_weight("t3", "t2", 1)
    graph.add_arc_weight("t3", "t4", 1)
    graph.add_arc_weight("t4", "t2", 1)
    graph.add_arc_weight("t4", "t3", 1)
    graph.add_arc_weight("t4", "t1", 1)

    # show graph
    print(graph.theme_graph)

    # get theme order
    order = graph.theme_order()
    print("order: ", order)
