import 'package:flutter/material.dart';
import 'package:supabase_flutter/supabase_flutter.dart';
import 'package:intl/intl.dart';
import 'package:snackly/screens/scan_screen.dart';

class MealTimelineScreen extends StatefulWidget {
  const MealTimelineScreen({Key? key}) : super(key: key);

  @override
  _MealTimelineScreenState createState() => _MealTimelineScreenState();
}

class _MealTimelineScreenState extends State<MealTimelineScreen>
    with TickerProviderStateMixin {
  Stream<List<Map<String, dynamic>>>? _mealStream;
  Map<String, List<Map<String, dynamic>>> _groupedMeals = {};
  
  // User profile data
  String? _username;
  double? _dailyCalorieGoalFromProfile;

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
      final event = data.event;
      if (event == AuthChangeEvent.signedIn ||
          event == AuthChangeEvent.userUpdated ||
          event == AuthChangeEvent.initialSession) {
        _fetchUserProfile();
        _setupMealStream();
      }
    });
  }

  void _initializeAnimations() {
    _fadeController = AnimationController(duration: const Duration(milliseconds: 800), vsync: this);
    _slideController = AnimationController(duration: const Duration(milliseconds: 600), vsync: this);
    _progressController = AnimationController(duration: const Duration(milliseconds: 1200), vsync: this);

    _fadeAnimation = Tween<double>(begin: 0.0, end: 1.0).animate(CurvedAnimation(parent: _fadeController, curve: Curves.easeInOut));
    _slideAnimation = Tween<Offset>(begin: const Offset(0, 0.3), end: Offset.zero).animate(CurvedAnimation(parent: _slideController, curve: Curves.easeOut));
    _progressAnimation = Tween<double>(begin: 0.0, end: 1.0).animate(CurvedAnimation(parent: _progressController, curve: Curves.elasticOut));

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
      if (mounted) setState(() => _username = null);
      return;
    }

    try {
      final data = await Supabase.instance.client
          .from('user_profiles')
          .select('username, daily_calorie_goal')
          .eq('id', user.id)
          .single();

      if (mounted) {
        setState(() {
          _username = data['username'] as String?;
          _dailyCalorieGoalFromProfile = (data['daily_calorie_goal'] as num?)?.toDouble();
        });
        _progressController.forward(from: 0.0);
      }
    } catch (e) {
      if (mounted) _showErrorSnackBar('Error fetching profile data.');
    }
  }

  void _setupMealStream() {
    final userId = Supabase.instance.client.auth.currentUser?.id;
    if (userId == null) {
      if (mounted) setState(() => _mealStream = Stream.value([]));
      return;
    }
    _mealStream = Supabase.instance.client
        .from('meal_logs')
        .stream(primaryKey: const ['id'])
        .eq('user_id', userId)
        .order('log_date', ascending: false)
        .order('created_at', ascending: false)
        .map((maps) => maps.map((map) => Map<String, dynamic>.from(map)).toList());
    if (mounted) setState(() {});
  }

  void _groupMeals(List<Map<String, dynamic>> meals) {
    _groupedMeals = {};
    for (var meal in meals) {
      final logDate = meal['log_date'] as String;
      (_groupedMeals[logDate] ??= []).add(meal);
    }
  }

  void _showErrorSnackBar(String message) {
    if (mounted) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text(message), backgroundColor: Colors.redAccent),
      );
    }
  }

  void _logout() async {
    try {
      await Supabase.instance.client.auth.signOut();
    } catch (e) {
      if (mounted) _showErrorSnackBar('Error logging out: ${e.toString()}');
    }
  }

  void _navigateToScanScreen() async {
    await Navigator.push<bool>(
      context,
      MaterialPageRoute(
        builder: (context) => ScanScreen(selectedDateForLog: DateTime.now()),
      ),
    );
  }

  Widget _buildWeeklySummary(Map<String, List<Map<String, dynamic>>> groupedMeals) {
    List<DateTime> last7Days = List.generate(7, (i) => DateTime.now().subtract(Duration(days: i))).reversed.toList();
    List<double> dailyCalories = last7Days.map((date) {
      String dateString = DateFormat('yyyy-MM-dd').format(date);
      return groupedMeals[dateString]?.fold(0.0, (sum, meal) => sum + ((meal['calories'] as num?)?.toDouble() ?? 0.0)) ?? 0.0;
    }).toList();

    double totalWeeklyCalories = dailyCalories.fold(0.0, (a, b) => a + b);
    double avgDailyCalories = dailyCalories.isEmpty ? 0 : totalWeeklyCalories / dailyCalories.length;
    double calorieGoal = _dailyCalorieGoalFromProfile ?? 2000;

    return SlideTransition(
      position: _slideAnimation,
      child: FadeTransition(
        opacity: _fadeAnimation,
        child: Container(
          margin: const EdgeInsets.all(16),
          padding: const EdgeInsets.all(20),
          decoration: BoxDecoration(
            gradient: const LinearGradient(
              begin: Alignment.topLeft,
              end: Alignment.bottomRight,
              colors: [Color(0xFFa1c4fd), Color(0xFFc2e9fb)],
            ),
            borderRadius: BorderRadius.circular(20),
            boxShadow: [BoxShadow(color: const Color(0xFFa1c4fd).withOpacity(0.3), blurRadius: 15, offset: const Offset(0, 5))],
          ),
          child: Column(
            children: [
              Text(
                'Weekly Overview',
                style: Theme.of(context).textTheme.headlineSmall?.copyWith(fontWeight: FontWeight.bold, color: const Color(0xFF3C486B)),
              ),
              const SizedBox(height: 20),
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceAround,
                children: [
                  _buildStatCard('🔥', 'Avg Daily', '${avgDailyCalories.toInt()}', 'kcal', Colors.orange),
                  _buildStatCard('📊', 'Weekly Total', '${totalWeeklyCalories.toInt()}', 'kcal', Colors.blue),
                  _buildStatCard('🎯', 'Goal', '${(calorieGoal > 0 ? (avgDailyCalories / calorieGoal) * 100 : 0).toInt()}%', 'of target', Colors.green),
                ],
              ),
              const SizedBox(height: 20),
              SizedBox(
                height: 100,
                child: Row(
                  mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                  crossAxisAlignment: CrossAxisAlignment.end,
                  children: last7Days.asMap().entries.map((entry) {
                    double calories = dailyCalories[entry.key];
                    double height = calorieGoal > 0 ? (calories / (calorieGoal * 1.2)).clamp(0.0, 1.0) * 80 : 0;
                    return _buildBar(date: entry.value, height: height, calories: calories, calorieGoal: calorieGoal);
                  }).toList(),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildBar({required DateTime date, required double height, required double calories, required double calorieGoal}) {
     Color barColor = Colors.red;
     if (calories > calorieGoal * 0.8) barColor = Colors.green;
     else if (calories > calorieGoal * 0.5) barColor = Colors.orange;

    return AnimatedBuilder(
      animation: _progressAnimation,
      builder: (context, child) => Column(
        mainAxisAlignment: MainAxisAlignment.end,
        children: [
          Container(
            width: 22,
            height: height * _progressAnimation.value,
            decoration: BoxDecoration(color: barColor, borderRadius: BorderRadius.circular(6)),
          ),
          const SizedBox(height: 4),
          Text(DateFormat('E').format(date), style: const TextStyle(fontSize: 12, color: Color(0xFF3C486B))),
        ],
      ),
    );
  }
  
  Widget _buildStatCard(String emoji, String label, String value, String unit, Color color) {
    return Column(
      children: [
        Text(emoji, style: const TextStyle(fontSize: 24)),
        const SizedBox(height: 4),
        Text(label, style: Theme.of(context).textTheme.labelMedium?.copyWith(color: const Color(0xFF3C486B))),
        Text(value, style: Theme.of(context).textTheme.titleMedium?.copyWith(fontWeight: FontWeight.bold, color: color)),
        Text(unit, style: Theme.of(context).textTheme.labelSmall?.copyWith(color: color.withOpacity(0.7))),
      ],
    );
  }

  Widget _buildTimelineItem(String date, List<Map<String, dynamic>> meals, bool isToday) {
    double totalCalories = meals.fold(0.0, (sum, meal) => sum + ((meal['calories'] as num?)?.toDouble() ?? 0.0));
    final displayDate = isToday ? 'Today' : DateFormat('MMM dd, yyyy').format(DateTime.parse(date));
    
    return IntrinsicHeight(
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          SizedBox(
            width: 50,
            child: Column(
              children: [
                Container(
                  width: 18, height: 18,
                  decoration: BoxDecoration(
                    color: isToday ? const Color(0xFFa1c4fd) : Colors.grey[300],
                    shape: BoxShape.circle,
                    border: Border.all(color: Colors.white, width: 3),
                  ),
                ),
                Expanded(child: Container(width: 2, color: Colors.grey[200])),
              ],
            ),
          ),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Padding(
                  padding: const EdgeInsets.only(top: 1.0),
                  child: Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      Text(displayDate, style: Theme.of(context).textTheme.titleLarge?.copyWith(fontWeight: FontWeight.bold, color: isToday ? const Color(0xFFa1c4fd) : const Color(0xFF3C486B))),
                      if (meals.isNotEmpty)
                        Container(
                          padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                          decoration: BoxDecoration(color: const Color(0xFFa1c4fd).withOpacity(0.2), borderRadius: BorderRadius.circular(12)),
                          child: Text('${totalCalories.toInt()} kcal', style: const TextStyle(fontWeight: FontWeight.bold, color: Color(0xFFa1c4fd), fontSize: 12)),
                        ),
                    ],
                  ),
                ),
                const SizedBox(height: 12),
                ...meals.map((meal) => _buildMealCard(meal)),
                const SizedBox(height: 24),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildMealCard(Map<String, dynamic> meal) {
    return Container(
      margin: const EdgeInsets.only(bottom: 12, right: 16),
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: Colors.white.withOpacity(0.85),
        borderRadius: BorderRadius.circular(12),
        boxShadow: [BoxShadow(color: Colors.black.withOpacity(0.05), blurRadius: 10, offset: const Offset(0, 4))],
      ),
      child: Row(
        children: [
          CircleAvatar(
            radius: 22,
            backgroundColor: const Color(0xFFc2e9fb),
            child: Text(
              (meal['food_name'] as String?)?.isNotEmpty == true ? (meal['food_name'] as String)[0].toUpperCase() : '?',
              style: const TextStyle(color: Color(0xFF3C486B), fontWeight: FontWeight.bold),
            ),
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(meal['food_name'] as String? ?? 'Unknown Food', style: const TextStyle(fontWeight: FontWeight.bold)),
                Text('C: ${meal['carbs']?.toStringAsFixed(1) ?? '-'}g • P: ${meal['protein']?.toStringAsFixed(1) ?? '-'}g • F: ${meal['fats']?.toStringAsFixed(1) ?? '-'}g', style: TextStyle(color: Colors.grey[600], fontSize: 12)),
              ],
            ),
          ),
          Text('${(meal['calories'] as num?)?.toStringAsFixed(0) ?? '0'}', style: const TextStyle(fontWeight: FontWeight.bold, color: Color(0xFFa1c4fd), fontSize: 16)),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    const Color darkTextColor = Color(0xFF3C486B);

    return Scaffold(
      extendBodyBehindAppBar: true,
      appBar: AppBar(
        title: Text(_username != null && _username!.isNotEmpty ? "$_username's Timeline" : 'Meal Timeline', style: const TextStyle(color: darkTextColor, fontWeight: FontWeight.bold)),
        backgroundColor: Colors.transparent,
        elevation: 0,
        iconTheme: const IconThemeData(color: darkTextColor),
        actions: [
          IconButton(icon: const Icon(Icons.logout), onPressed: _logout, tooltip: 'Logout'),
        ],
      ),
      body: Container(
        decoration: BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topCenter, end: Alignment.bottomCenter,
            colors: [const Color(0xFFE0F7FA).withOpacity(0.5), Colors.white],
          ),
        ),
        child: SafeArea(
          child: StreamBuilder<List<Map<String, dynamic>>>(
            stream: _mealStream,
            builder: (context, snapshot) {
              if (snapshot.connectionState == ConnectionState.waiting && !snapshot.hasData) {
                return const Center(child: CircularProgressIndicator());
              }
              if (snapshot.hasError) return Center(child: Text('Error: ${snapshot.error}'));
              
              final allMeals = snapshot.data ?? [];
              _groupMeals(allMeals);
              List<String> sortedDates = _groupedMeals.keys.toList()..sort((a, b) => b.compareTo(a));

              return CustomScrollView(
                slivers: [
                  SliverToBoxAdapter(child: _buildWeeklySummary(_groupedMeals)),
                  if (sortedDates.isEmpty)
                    SliverFillRemaining(
                      child: Center(
                        child: Text("No meals logged yet.", style: TextStyle(color: Colors.grey[600]))
                      )
                    )
                  else
                    SliverList(
                      delegate: SliverChildBuilderDelegate(
                        (context, index) {
                          String date = sortedDates[index];
                          bool isToday = date == DateFormat('yyyy-MM-dd').format(DateTime.now());
                          return _buildTimelineItem(date, _groupedMeals[date] ?? [], isToday);
                        },
                        childCount: sortedDates.length,
                      ),
                    ),
                ],
              );
            },
          ),
        ),
      ),
      floatingActionButton: ScaleTransition(
        scale: _fadeAnimation,
        child: FloatingActionButton.extended(
          onPressed: _navigateToScanScreen,
          tooltip: 'Scan Food',
          icon: const Icon(Icons.camera_alt, color: Colors.white),
          label: const Text('Scan Food', style: TextStyle(color: Colors.white, fontWeight: FontWeight.bold)),
          backgroundColor: const Color(0xFF81C784), // Pastel Green
        ),
      ),
    );
  }
}