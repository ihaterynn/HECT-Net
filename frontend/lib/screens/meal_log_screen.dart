import 'package:flutter/material.dart';
import 'package:supabase_flutter/supabase_flutter.dart';
import 'package:intl/intl.dart';
import 'package:snackly/screens/scan_screen.dart';

class MealLogScreen extends StatefulWidget {
  const MealLogScreen({Key? key}) : super(key: key);

  @override
  _MealLogScreenState createState() => _MealLogScreenState();
}

class _MealLogScreenState extends State<MealLogScreen>
    with TickerProviderStateMixin {
  Stream<List<Map<String, dynamic>>>? _mealStream;
  DateTime _selectedDate = DateTime.now();
  Map<String, List<Map<String, dynamic>>> _groupedMeals = {};

  // User profile data state variables
  String? _username;
  double? _dailyCalorieGoalFromProfile;
  String? _fitnessGoalText;
  double? _heightCm;
  double? _currentBodyWeightKg;

  // Animation controllers
  late AnimationController _fadeController;
  late AnimationController _slideController;
  late AnimationController _progressController;
  late Animation<double> _fadeAnimation;
  late Animation<Offset> _slideAnimation;
  late Animation<double> _progressAnimation;

  @override
  void initState() {
    super.initState();
    _initializeAnimations();
    _fetchUserProfile();
    _setupMealStream();

    Supabase.instance.client.auth.onAuthStateChange.listen((data) {
      final AuthChangeEvent event = data.event;
      if (event == AuthChangeEvent.signedIn ||
          event == AuthChangeEvent.userUpdated ||
          event == AuthChangeEvent.initialSession) {
        _fetchUserProfile();
        _setupMealStream();
      }
    });
  }

  void _initializeAnimations() {
    _fadeController = AnimationController(
      duration: const Duration(milliseconds: 800),
      vsync: this,
    );
    _slideController = AnimationController(
      duration: const Duration(milliseconds: 600),
      vsync: this,
    );
    _progressController = AnimationController(
      duration: const Duration(milliseconds: 1200),
      vsync: this,
    );

    _fadeAnimation = Tween<double>(begin: 0.0, end: 1.0).animate(
      CurvedAnimation(parent: _fadeController, curve: Curves.easeInOut),
    );
    _slideAnimation = Tween<Offset>(
      begin: const Offset(0, 0.3),
      end: Offset.zero,
    ).animate(CurvedAnimation(parent: _slideController, curve: Curves.easeOut));
    _progressAnimation = Tween<double>(begin: 0.0, end: 1.0).animate(
      CurvedAnimation(parent: _progressController, curve: Curves.elasticOut),
    );

    _fadeController.forward();
    _slideController.forward();
  }

  @override
  void dispose() {
    _fadeController.dispose();
    _slideController.dispose();
    _progressController.dispose();
    super.dispose();
  }

  Future<void> _fetchUserProfile() async {
    final user = Supabase.instance.client.auth.currentUser;
    if (user == null) {
      if (mounted) {
        setState(() {
          _username = null;
          _dailyCalorieGoalFromProfile = null;
          _fitnessGoalText = null;
          _heightCm = null;
          _currentBodyWeightKg = null;
        });
      }
      return;
    }

    try {
      final data = await Supabase.instance.client
          .from('user_profiles')
          .select(
              'username, daily_calorie_goal, fitness_goal, height_cm, current_body_weight_kg')
          .eq('id', user.id)
          .single();

      if (mounted) {
        setState(() {
          _username = data['username'] as String?;
          _dailyCalorieGoalFromProfile =
              (data['daily_calorie_goal'] as num?)?.toDouble();
          _fitnessGoalText = data['fitness_goal'] as String?;
          _heightCm = (data['height_cm'] as num?)?.toDouble();
          _currentBodyWeightKg =
              (data['current_body_weight_kg'] as num?)?.toDouble();
        });
        _progressController.forward();
      }
    } catch (e) {
      if (mounted) {
        print("Error fetching profile in MealLogScreen: ${e.toString()}");
        _showErrorSnackBar(
            'Error fetching profile data. Please ensure your profile is fully set up.');
      }
    }
  }

  void _setupMealStream() {
    final userId = Supabase.instance.client.auth.currentUser?.id;
    if (userId == null) {
      if (mounted) {
        setState(() {
          _mealStream = Stream.value([]);
        });
      }
      return;
    }
    _mealStream = Supabase.instance.client
        .from('meal_logs')
        .stream(primaryKey: const ['id'])
        .eq('user_id', userId)
        .order('log_date', ascending: false)
        .order('created_at', ascending: false)
        .map((maps) =>
            maps.map((map) => Map<String, dynamic>.from(map)).toList());
    if (mounted) {
      setState(() {});
    }
  }

  void _groupMeals(List<Map<String, dynamic>> meals) {
    _groupedMeals = {};
    for (var meal in meals) {
      final logDate = meal['log_date'] as String;
      if (_groupedMeals.containsKey(logDate)) {
        _groupedMeals[logDate]!.add(meal);
      } else {
        _groupedMeals[logDate] = [meal];
      }
    }
  }

  Future<void> _selectDate(BuildContext context) async {
    final DateTime? picked = await showDatePicker(
      context: context,
      initialDate: _selectedDate,
      firstDate: DateTime(2000),
      lastDate: DateTime(2101),
    );
    if (picked != null && picked != _selectedDate) {
      if (mounted) {
        setState(() {
          _selectedDate = picked;
        });
      }
    }
  }

  void _logout() async {
    try {
      await Supabase.instance.client.auth.signOut();
    } catch (e) {
      if (mounted) _showErrorSnackBar('Error logging out: ${e.toString()}');
    }
  }

  void _showErrorSnackBar(String message) {
    if (mounted) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
            content: Text(message),
            backgroundColor: Theme.of(context).colorScheme.error),
      );
    }
  }

  void _navigateToScanScreen() async {
    await Navigator.push<bool>(
      context,
      MaterialPageRoute(
          builder: (context) => ScanScreen(
                selectedDateForLog: _selectedDate,
              )),
    );
  }

  Future<void> _deleteMeal(int mealId, String? foodName) async {
    try {
      await Supabase.instance.client
          .from('meal_logs')
          .delete()
          .eq('id', mealId);

      _setupMealStream();

      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('Meal deleted successfully!'),
            backgroundColor: const Color(0xFF59CE8F),
            behavior: SnackBarBehavior.floating,
            action: SnackBarAction(
              label: 'Undo',
              onPressed: () {
                ScaffoldMessenger.of(context).showSnackBar(
                  const SnackBar(
                    content: Text('Undo feature coming soon!'),
                    duration: Duration(seconds: 2),
                  ),
                );
              },
            ),
          ),
        );
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('Failed to delete meal: ${e.toString()}'),
            backgroundColor: const Color(0xFFE57373),
            behavior: SnackBarBehavior.floating,
          ),
        );
      }
    }
  }

  Future<void> _showDeleteConfirmation(int mealId, String? foodName) async {
    final bool? shouldDelete = await showDialog<bool>(
      context: context,
      builder: (BuildContext context) {
        return AlertDialog(
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
          title: Row(
            children: [
              Icon(Icons.warning_amber_rounded,
                  color: const Color(0xFFFFB74D), size: 28),
              const SizedBox(width: 8),
              const Text('Delete Meal'),
            ],
          ),
          content: Text(
            'Are you sure you want to delete "${foodName ?? 'this meal'}"?\n\nThis action cannot be undone.',
            style: Theme.of(context).textTheme.bodyMedium,
          ),
          actions: [
            TextButton(
              onPressed: () => Navigator.of(context).pop(false),
              child: Text(
                'Cancel',
                style: TextStyle(color: Color(0xFF1976D2)), // Changed from theme primary to darker blue
              ),
            ),
            ElevatedButton(
              onPressed: () => Navigator.of(context).pop(true),
              style: ElevatedButton.styleFrom(
                backgroundColor: Colors.red,
                foregroundColor: Colors.white,
                shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(8)),
              ),
              child: const Text('Delete'),
            ),
          ],
        );
      },
    );

    if (shouldDelete == true) {
      await _deleteMeal(mealId, foodName);
    }
  }

  Widget _buildNutritionSummary(
      List<Map<String, dynamic>> mealsForSelectedDate) {
    double totalCalories = mealsForSelectedDate.fold(0.0, (sum, meal) {
      return sum + ((meal['calories'] as num?)?.toDouble() ?? 0.0);
    });
    double totalCarbs = mealsForSelectedDate.fold(0.0, (sum, meal) {
      return sum + ((meal['carbs'] as num?)?.toDouble() ?? 0.0);
    });
    double totalProtein = mealsForSelectedDate.fold(0.0, (sum, meal) {
      return sum + ((meal['protein'] as num?)?.toDouble() ?? 0.0);
    });
    double totalFats = mealsForSelectedDate.fold(0.0, (sum, meal) {
      return sum + ((meal['fats'] as num?)?.toDouble() ?? 0.0);
    });

    double calorieGoal = _dailyCalorieGoalFromProfile ?? 2000;
    double calorieProgress = (totalCalories / calorieGoal).clamp(0.0, 1.0);

    Color getCalorieProgressColor(double progress) {
      if (progress < 0.3) return Colors.red;
      if (progress < 0.5) return Colors.orange;
      if (progress < 0.7) return Colors.yellow[700]!;
      if (progress < 0.9) return Colors.lightGreen;
      return Colors.green;
    }

    return SlideTransition(
      position: _slideAnimation,
      child: FadeTransition(
        opacity: _fadeAnimation,
        child: Container(
          margin: const EdgeInsets.all(16),
          padding: const EdgeInsets.all(20),
          decoration: BoxDecoration(
            gradient: LinearGradient(
              begin: Alignment.topLeft,
              end: Alignment.bottomRight,
              colors: [
                Theme.of(context).colorScheme.primaryContainer.withOpacity(0.8),
                Theme.of(context).colorScheme.secondaryContainer.withOpacity(0.6),
              ],
            ),
            borderRadius: BorderRadius.circular(20),
            boxShadow: [
              BoxShadow(
                color: Colors.black.withOpacity(0.1),
                blurRadius: 10,
                offset: const Offset(0, 4),
              ),
            ],
          ),
          child: Column(
            children: [
              Text(
                'Today\'s Progress',
                style: Theme.of(context).textTheme.headlineSmall?.copyWith(
                      fontWeight: FontWeight.bold,
                      color: const Color(0xFF1976D2), // Darker blue text
                    ),
              ),
              const SizedBox(height: 20),
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                children: [
                  // Calories with circular progress
                  Column(
                    children: [
                      AnimatedBuilder(
                        animation: _progressAnimation,
                        builder: (context, child) {
                          return SizedBox(
                            width: 80,
                            height: 80,
                            child: Stack(
                              children: [
                                CircularProgressIndicator(
                                  value: calorieProgress * _progressAnimation.value,
                                  strokeWidth: 6,
                                  backgroundColor: Colors.grey[300],
                                  valueColor: AlwaysStoppedAnimation<Color>(
                                    getCalorieProgressColor(calorieProgress),
                                  ),
                                ),
                                Center(
                                  child: Column(
                                    mainAxisAlignment: MainAxisAlignment.center,
                                    children: [
                                      Text(
                                        '🔥',
                                        style: TextStyle(fontSize: 20),
                                      ),
                                      Text(
                                        '${(totalCalories * _progressAnimation.value).toInt()}',
                                        style: Theme.of(context)
                                            .textTheme
                                            .labelSmall
                                            ?.copyWith(
                                              fontWeight: FontWeight.bold,
                                            ),
                                      ),
                                    ],
                                  ),
                                ),
                              ],
                            ),
                          );
                        },
                      ),
                      const SizedBox(height: 8),
                      Text(
                        'Calories',
                        style: Theme.of(context).textTheme.labelMedium,
                      ),
                      Text(
                        '${totalCalories.toInt()}/${calorieGoal.toInt()}',
                        style: Theme.of(context).textTheme.labelSmall,
                      ),
                    ],
                  ),
                  // Carbs
                  _buildNutrientColumn(
                    'assets/images/bread.png',
                    'Carbs',
                    totalCarbs,
                    'g',
                    Colors.orange,
                  ),
                  // Protein
                  _buildNutrientColumn(
                    'assets/images/steak.png',
                    'Protein',
                    totalProtein,
                    'g',
                    Colors.red,
                  ),
                  // Fats
                  _buildNutrientColumn(
                    'assets/images/milk2.png',
                    'Fats',
                    totalFats,
                    'g',
                    Colors.blue,
                  ),
                ],
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildNutrientColumn(
    String assetPath,
    String label,
    double value,
    String unit,
    Color color,
  ) {
    return AnimatedBuilder(
      animation: _progressAnimation,
      builder: (context, child) {
        return Column(
          children: [
            Container(
              width: 60,
              height: 60,
              decoration: BoxDecoration(
                color: color.withOpacity(0.1),
                borderRadius: BorderRadius.circular(15),
                border: Border.all(
                  color: color.withOpacity(0.3),
                  width: 2,
                ),
              ),
              child: Center(
                child: Image.asset(
                  assetPath,
                  width: 30,
                  height: 30,
                  errorBuilder: (context, error, stackTrace) {
                    return Icon(
                      Icons.fastfood,
                      color: color,
                      size: 30,
                    );
                  },
                ),
              ),
            ),
            const SizedBox(height: 8),
            Text(
              label,
              style: Theme.of(context).textTheme.labelMedium,
            ),
            Text(
              '${(value * _progressAnimation.value).toStringAsFixed(1)}$unit',
              style: Theme.of(context).textTheme.labelSmall?.copyWith(
                    fontWeight: FontWeight.bold,
                    color: color,
                  ),
            ),
          ],
        );
      },
    );
  }

  @override
  Widget build(BuildContext context) {
    String selectedDateString = DateFormat('yyyy-MM-dd').format(_selectedDate);

    return Scaffold(
      appBar: AppBar(
        title: Text(_username != null && _username!.isNotEmpty
            ? "$_username's Meal Log"
            : 'My Meal Log'),
        actions: [
          IconButton(
            icon: const Icon(Icons.calendar_today),
            onPressed: () => _selectDate(context),
            tooltip: 'Select Date',
          ),
          IconButton(
            icon: const Icon(Icons.person_outline),
            onPressed: () {
              showDialog(
                context: context,
                builder: (context) => AlertDialog(
                  title: const Text('User Profile Summary'),
                  content: SingleChildScrollView(
                    child: ListBody(
                      children: <Widget>[
                        Text(
                            'Email: ${Supabase.instance.client.auth.currentUser?.email ?? 'N/A'}'),
                        Text('Username: ${_username ?? 'Not set'}'),
                        Text('Fitness Goal: ${_fitnessGoalText ?? 'Not set'}'),
                        Text(
                            'Daily Calorie Goal: ${_dailyCalorieGoalFromProfile?.toStringAsFixed(0) ?? 'Not set'} Cal'),
                        Text(
                            'Height: ${_heightCm?.toStringAsFixed(1) ?? 'Not set'} cm'),
                        Text(
                            'Current Weight: ${_currentBodyWeightKg?.toStringAsFixed(1) ?? 'Not set'} kg'),
                      ],
                    ),
                  ),
                  actions: <Widget>[
                    TextButton(
                      child: const Text('Edit Profile'),
                      onPressed: () {
                        Navigator.of(context).pop();
                        ScaffoldMessenger.of(context).showSnackBar(
                            const SnackBar(
                                content: Text(
                                    'Navigate to full profile editing screen.')));
                      },
                    ),
                    TextButton(
                      child: const Text('Close'),
                      onPressed: () => Navigator.of(context).pop(),
                    ),
                  ],
                ),
              );
            },
            tooltip: 'View Profile',
          ),
          IconButton(
            icon: const Icon(Icons.logout),
            onPressed: _logout,
            tooltip: 'Logout',
          ),
        ],
      ),
      body: StreamBuilder<List<Map<String, dynamic>>>(
        stream: _mealStream,
        builder: (context, snapshot) {
          if (snapshot.connectionState == ConnectionState.waiting &&
              _mealStream == null) {
            return const Center(child: Text("Initializing..."));
          }
          if (snapshot.connectionState == ConnectionState.waiting &&
              !snapshot.hasData) {
            return const Center(child: CircularProgressIndicator());
          }
          if (snapshot.hasError) {
            return Center(
                child: Text('Error loading meals: ${snapshot.error}'));
          }

          final allMeals = snapshot.data ?? [];
          _groupMeals(allMeals);

          final mealsForSelectedDate = _groupedMeals[selectedDateString] ?? [];

          return Column(
            children: [
              // Enhanced nutrition summary section
              _buildNutritionSummary(mealsForSelectedDate),
              
              // Date header
              Padding(
                padding: const EdgeInsets.symmetric(horizontal: 16.0),
                child: Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [
                    Text(
                      DateFormat.yMMMMd().format(_selectedDate),
                      style: Theme.of(context)
                          .textTheme
                          .headlineSmall
                          ?.copyWith(fontWeight: FontWeight.bold),
                    ),
                    Text(
                      '${mealsForSelectedDate.length} meals',
                      style: Theme.of(context).textTheme.labelMedium?.copyWith(
                            color: Color(0xFF1976D2), // Changed from theme primary to darker blue
                          ),
                    ),
                  ],
                ),
              ),
              const SizedBox(height: 16),
              
              // Meals list
              if (mealsForSelectedDate.isEmpty)
                Expanded(
                  child: Center(
                    child: FadeTransition(
                      opacity: _fadeAnimation,
                      child: Padding(
                        padding: const EdgeInsets.all(20.0),
                        child: Column(
                          mainAxisAlignment: MainAxisAlignment.center,
                          children: [
                            Icon(
                              Icons.restaurant_menu,
                              size: 64,
                              color: Colors.grey[400],
                            ),
                            const SizedBox(height: 16),
                            Text(
                              'No meals logged for ${DateFormat.yMMMMd().format(_selectedDate)}.',
                              textAlign: TextAlign.center,
                              style: Theme.of(context)
                                  .textTheme
                                  .titleMedium
                                  ?.copyWith(color: Colors.grey[600]),
                            ),
                            const SizedBox(height: 8),
                            Text(
                              'Tap the camera button below to add a meal.',
                              textAlign: TextAlign.center,
                              style: Theme.of(context)
                                  .textTheme
                                  .bodyMedium
                                  ?.copyWith(color: Colors.grey[500]),
                            ),
                          ],
                        ),
                      ),
                    ),
                  ),
                )
              else
                Expanded(
                  child: ListView.builder(
                    itemCount: mealsForSelectedDate.length,
                    itemBuilder: (context, index) {
                      final meal = mealsForSelectedDate[index];
                      return SlideTransition(
                        position: Tween<Offset>(
                          begin: Offset(1.0, 0.0),
                          end: Offset.zero,
                        ).animate(CurvedAnimation(
                          parent: _slideController,
                          curve: Interval(
                            index * 0.1,
                            (index * 0.1) + 0.3,
                            curve: Curves.easeOut,
                          ),
                        )),
                        child: Dismissible(
                          key: Key('meal_${meal['id']}'),
                          direction: DismissDirection.endToStart,
                          background: Container(
                            alignment: Alignment.centerRight,
                            padding: const EdgeInsets.only(right: 20),
                            margin: const EdgeInsets.symmetric(
                                horizontal: 16, vertical: 6),
                            decoration: BoxDecoration(
                              color: Colors.red,
                              borderRadius: BorderRadius.circular(12),
                            ),
                            child: const Icon(
                              Icons.delete_forever,
                              color: Colors.white,
                              size: 32,
                            ),
                          ),
                          confirmDismiss: (direction) async {
                            return await showDialog(
                              context: context,
                              builder: (BuildContext context) {
                                return AlertDialog(
                                  title: const Text('Delete Meal'),
                                  content: Text(
                                      'Are you sure you want to delete "${meal['food_name']}"?'),
                                  actions: [
                                    TextButton(
                                      onPressed: () =>
                                          Navigator.of(context).pop(false),
                                      child: const Text('Cancel'),
                                    ),
                                    TextButton(
                                      onPressed: () =>
                                          Navigator.of(context).pop(true),
                                      style: TextButton.styleFrom(
                                          foregroundColor: Colors.red),
                                      child: const Text('Delete'),
                                    ),
                                  ],
                                );
                              },
                            );
                          },
                          onDismissed: (direction) async {
                            await _deleteMeal(meal['id'], meal['food_name']);
                          },
                          child: Card(
                            margin: const EdgeInsets.symmetric(
                                horizontal: 16, vertical: 6),
                            elevation: 2,
                            shape: RoundedRectangleBorder(
                                borderRadius: BorderRadius.circular(12)),
                            child: ListTile(
                              contentPadding: const EdgeInsets.symmetric(
                                  vertical: 10, horizontal: 16),
                              leading: Hero(
                                tag: 'meal_${meal['id']}',
                                child: CircleAvatar(
                                  backgroundColor: Color(0xFFE3F2FD), // Changed from primaryContainer to pastel blue
                                  child: Text(
                                    (meal['food_name'] as String?)
                                                ?.isNotEmpty ==
                                            true
                                        ? (meal['food_name'] as String)[0]
                                            .toUpperCase()
                                        : 'X',
                                    style: TextStyle(
                                      color: Color(0xFF1976D2), // Changed from onPrimaryContainer to darker blue
                                      fontWeight: FontWeight.bold,
                                    ),
                                  ),
                                ),
                              ),
                              title: Text(
                                meal['food_name'] as String? ?? 'Unknown Food',
                                style: Theme.of(context)
                                    .textTheme
                                    .titleMedium
                                    ?.copyWith(fontWeight: FontWeight.w600),
                              ),
                              subtitle: Text(
                                  'Carbs: ${meal['carbs']?.toStringAsFixed(1) ?? '-'}g, Prot: ${meal['protein']?.toStringAsFixed(1) ?? '-'}g, Fat: ${meal['fats']?.toStringAsFixed(1) ?? '-'}g'),
                              trailing: Row(
                                mainAxisSize: MainAxisSize.min,
                                children: [
                                  Text(
                                    '${(meal['calories'] as num?)?.toStringAsFixed(0) ?? '0'} Cal',
                                    style: Theme.of(context)
                                        .textTheme
                                        .titleMedium
                                        ?.copyWith(
                                          fontWeight: FontWeight.bold,
                                          color: Theme.of(context)
                                              .colorScheme
                                              .primary,
                                        ),
                                  ),
                                  const SizedBox(width: 8),
                                  IconButton(
                                    icon: const Icon(Icons.delete_outline,
                                        color: Colors.red),
                                    onPressed: () => _showDeleteConfirmation(
                                        meal['id'], meal['food_name']),
                                    tooltip: 'Delete meal',
                                  ),
                                ],
                              ),
                              onTap: () {
                                print("Tapped on meal: ${meal['food_name']}");
                              },
                            ),
                          ),
                        ),
                      );
                    },
                  ),
                ),
            ],
          );
        },
      ),
      floatingActionButton: Container(
        decoration: BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topLeft,
            end: Alignment.bottomRight,
            colors: [
              Color(0xFFE3F2FD).withOpacity(0.8), // Changed from Theme.of(context).colorScheme.primaryContainer to pastel blue
              Colors.white.withOpacity(0.6), // Changed from Theme.of(context).colorScheme.secondaryContainer to white
            ],
          ),
          borderRadius: BorderRadius.circular(20),
          boxShadow: [
            BoxShadow(
              color: Colors.black.withOpacity(0.1),
              blurRadius: 10,
              offset: const Offset(0, 4),
            ),
          ],
        ),
        child: FloatingActionButton.extended(
          onPressed: _navigateToScanScreen,
          tooltip: 'Scan Food',
          icon: const Icon(Icons.camera_alt),
          label: const Text('Scan Food'),
          backgroundColor: const Color.fromARGB(255, 90, 192, 99), // Pastel green
          foregroundColor: const Color.fromARGB(255, 255, 255, 255), // Dark green text and icon
          elevation: 0, // Remove default elevation since we have custom shadow
        ),
      ),
    );
  }
}