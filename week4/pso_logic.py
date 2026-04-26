
import random
import math

class Particle:
    """ har tanolkayak xaleky julaua lasar shahshaka"""
  

    def __init__(self, x, y, color):
        """ auayn point x y color har yakayan ja goal yan har pointek auayh constuctior"""
       
        self.x = x                          # current x position
        self.y = y                          # current y position
        self.vx = random.uniform(-2, 2)
        """ particle sarata randomly dagore""" 
            #auayn vilocity x akaya wata xery range -2 bo 2 agar negative bo posative
        self.vy = random.uniform(-2, 2)
            # auayan veilocity yakaya
        self.best_x = x                    
        self.best_y = y     
        """ au x y lagal dozynauay sheuana nueyaka nue dakreta"""                
        self.best_fitness = float('inf') 

           #  wata la zhmary gawara dast pe dakachand la amanj nzytka ta point bchuktr by bashtra infity
           #wata har maudayaky rastaqna bchuk be updated dakretuaa
        self.color = color      
                    

    def fitness(self, goal_x, goal_y):
        # auayan calucaltey fintess ba ecludance distance
       
        """ ba kurty fitness wata charasaraka chand bash"""

        return math.sqrt((self.x - goal_x) ** 2 + (self.y - goal_y) ** 2)



    def update_personal_best(self, goal_x, goal_y):
        #aya pegay estam bashtera yan pesh 
         #wata check  daka  aya pegay estam basthra lauany peshu agar basthra bbest bka aua
         # waku balnday bena pesh chaut
       
        f = self.fitness(goal_x, goal_y)
        """ au desginacy atu esta ley kamay bchuk tra au bashtar"""
        if f < self.best_fitness:
            """ wata au pointeky essta ley bashtr bu lauay peshutre"""
            self.best_fitness = f
            self.best_x = self.x
            self.best_y = self.y
            """ save new best poistion best fitness function daka"""
        return f
    

# updated xeray la dahtu bo chue bjuleny ba xeray
    def update_velocity(self, global_best_x, global_best_y, w, c1, c2):
        """ bashtryn shuen lalayan hamu tanolkakanaua
        w zabr c1 factary ferbuny kasyu auay tr factary ferbuny komalayty
        bryary dada chon bjule"""
        
        r1 = random.random()
        r2 = random.random()
        """ au random number between zero and one"""
        
        # Update x-velocity
        self.vx = (w * self.vx
                   + c1 * r1 * (self.best_x - self.x)
                   + c2 * r2 * (global_best_x - self.x))
        """ agar w barz bu bardauam dabe la royshtn
        agar lauaz bu xau dabetua party duam bashtryn pagey ka peshu ley bua labirytay"""
        
        # Update y-velocity
        self.vy = (w * self.vy
                   + c1 * r1 * (self.best_y - self.y)
                   + c2 * r2 * (global_best_y - self.y))

    def update_position(self):
        """ auany tanolkaka dabata julakaya nwe"""
        """
        Move the particle by adding velocity to current position.
        
        x_new = x + v
        y_new = y + v
        """
        self.x += self.vx
        self.y += self.vy
    
    def get_speed(self):
        """ wata tanlokaka chand xera dajuey"""
        """Calculate the magnitude of velocity (particle speed)."""
        return math.sqrt(self.vx ** 2 + self.vy ** 2)


def run_pso_step(particles, goal_x, goal_y, global_best, w, c1, c2):
    #wata hamu functianak ka run dabe yak iteration
    """ contorly hamu iterationakan daka ka lanau programaka ru dadad """
  
    gbx, gby, gbf = global_best

    # ---- Step 1 & 2: Evaluate fitness and update personal bests ----
    for p in particles:
        """ banau fitnesaakan darua wata calculaty distand update"""
        f = p.update_personal_best(goal_x, goal_y)
        # Update global best if this particle found a better solution
        if f < gbf:
            gbx, gby, gbf = p.x, p.y, f

    # ---- Step 3 & 4: Update velocities and positions ----
    for p in particles:
        p.update_velocity(gbx, gby, w, c1, c2)
        p.update_position()

    return (gbx, gby, gbf)


"""بیرۆکەیەکی سادە (زۆر گرنگە)

تەنۆلکەیەک (باڵندە) لەسەر بنەمای ٣ شت دەجوڵێت: ١.

🧭 جوڵەی ئێستای (زەبری)
🧠 باشترین پێگەی خۆی (بیرەوەری)
🌍 باشترین شوێن کە هەموو تەنۆلکەکان دەیدۆزنەوە"""









