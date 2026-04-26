
import random
import math

""" ba kurty aua project bo aua bakar det chon bashtyrn charasar bdozyau ka peshtr nayznay
nmuna waku bashtrkrndy reuray gayandn for example amazon"""
class Particle:
    """
    au class mabsty tanokakana
    """

    def __init__(self, x, y, color):
        """ auayn point x y color har yakayan ja goal yan har pointek auayh constuctior"""
       
        self.x = x                          # current x position
        self.y = y                          # current y position
        self.vx = random.uniform(-2, 2) 
            #auayn vilocity x akaya wata xery range -2 bo 2 agar negative bo posative
        self.vy = random.uniform(-2, 2)
            # auayan veilocity yakaya
        self.best_x = x                    
        self.best_y = y                     
        self.best_fitness = float('inf') 
           # chand la amanj nzytka ta point bchuktr by bashtra infity
           #wata har maudayaky rastaqna bchuk be updated dakretuaa
        self.color = color      
                    

    def fitness(self, goal_x, goal_y):
        # auayan calucaltey fintess ba ecludance distance
        """
        Calculate fitness as Euclidean distance to goal.
        
        In PSO, we minimize fitness (lower is better).
        Fitness = sqrt((x - goal_x)^2 + (y - goal_y)^2)
        
        Args:
            goal_x, goal_y: Target coordinates in 2D space
            
        Returns:
            Fitness value (distance to goal, lower is better)
        """

        return math.sqrt((self.x - goal_x) ** 2 + (self.y - goal_y) ** 2)



    def update_personal_best(self, goal_x, goal_y):
        #aya pegay estam bashtera yan pesh
        """
        Check if current position is better than the particle's personal best.
        If so, update the personal best position and fitness.
        
        Args:
            goal_x, goal_y: Target coordinates
            
        Returns:
            Current fitness value
        """
        f = self.fitness(goal_x, goal_y)
        """ au desginacy atu esta ley"""
        if f < self.best_fitness:
            """ wata au pointeky essta ley bashtr bu lauay peshutre"""
            self.best_fitness = f
            self.best_x = self.x
            self.best_y = self.y
            """ save new best poistion best fitness function daka"""
        return f

    def update_velocity(self, global_best_x, global_best_y, w, c1, c2):
        """ bashtryn shuen lalayan hamu tanolkakanaua
        w zabr c1 factary ferbuny kasyu auay tr factary ferbuny komalayty
        bryary dada chon bjule"""
        """
        Update particle velocity using the PSO velocity formula:
        
        v_new = w*v_old + c1*r1*(pBest - x) + c2*r2*(gBest - x)
        
        This formula balances three forces:
        1. Inertia (w*v_old): Keep moving in the same direction
        2. Cognitive (c1*r1*(pBest-x)): Pull toward personal best
        3. Social (c2*r2*(gBest-x)): Pull toward global best
        
        Args:
            global_best_x, global_best_y: Best position found by swarm
            w: Inertia weight (0.0 - 1.0) — higher values = more momentum
            c1: Cognitive coefficient (typically 1.5) — pull toward personal best
            c2: Social coefficient (typically 1.5) — pull toward global best
        """
        # Random values for stochastic exploration
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


""" simple difintion
auaya ka projectaka sarata  randomly dajule
dautar la pointkany traua fer dabe
distance mabasty fitness
personal best aya era bashtrn shuena agar haua print kay"""