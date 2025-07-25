import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'dart:io'; // Required for File type
import 'package:supabase_flutter/supabase_flutter.dart'; // Import Supabase

class ProfileScreen extends StatefulWidget {
  const ProfileScreen({Key? key}) : super(key: key);

  @override
  _ProfileScreenState createState() => _ProfileScreenState();
}

class _ProfileScreenState extends State<ProfileScreen> with TickerProviderStateMixin {
  final _formKey = GlobalKey<FormState>();
  bool _loading = true; 

  // Animation controllers
  late AnimationController _fadeController;
  late AnimationController _slideController;
  late AnimationController _scaleController;
  late Animation<double> _fadeAnimation;
  late Animation<Offset> _slideAnimation;
  late Animation<double> _scaleAnimation;

  // User data fields
  String? _userEmail;
  String _username = '';
  double _weight = 70.0; 
  double _height = 170.0;
  String _fitnessGoal = 'maintain';
  double _calculatedCalories = 2000.0;
  File? _profileImage;
  final ImagePicker _picker = ImagePicker();

  // Controllers
  late TextEditingController _usernameController;
  late TextEditingController _weightController;
  late TextEditingController _heightController;

  @override
  void initState() {
    super.initState();
    _initializeAnimations();
    _usernameController = TextEditingController(text: _username);
    _weightController = TextEditingController(text: _weight.toString());
    _heightController = TextEditingController(text: _height.toString());
    _loadProfileData();
  }

  void _initializeAnimations() {
    _fadeController = AnimationController(
      duration: const Duration(milliseconds: 1000),
      vsync: this,
    );
    _slideController = AnimationController(
      duration: const Duration(milliseconds: 800),
      vsync: this,
    );
    _scaleController = AnimationController(
      duration: const Duration(milliseconds: 600),
      vsync: this,
    );

    _fadeAnimation = Tween<double>(begin: 0.0, end: 1.0).animate(
      CurvedAnimation(parent: _fadeController, curve: Curves.easeInOut),
    );
    _slideAnimation = Tween<Offset>(
      begin: const Offset(0, 0.3),
      end: Offset.zero,
    ).animate(CurvedAnimation(parent: _slideController, curve: Curves.easeOut));
    _scaleAnimation = Tween<double>(begin: 0.8, end: 1.0).animate(
      CurvedAnimation(parent: _scaleController, curve: Curves.elasticOut),
    );

    _fadeController.forward();
    _slideController.forward();
    _scaleController.forward();
  }

  @override
  void dispose() {
    _fadeController.dispose();
    _slideController.dispose();
    _scaleController.dispose();
    _usernameController.dispose();
    _weightController.dispose();
    _heightController.dispose();
    super.dispose();
  }

  Future<void> _loadProfileData() async {
    if (!mounted) return;
    setState(() => _loading = true);

    final user = Supabase.instance.client.auth.currentUser;
    if (user == null) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('User not logged in. Cannot load profile.'), backgroundColor: Colors.red),
        );
        setState(() => _loading = false);
      }
      return;
    }

    _userEmail = user.email;

    try {
      final profileData = await Supabase.instance.client
          .from('user_profiles')
          .select('username, current_body_weight_kg, height_cm, fitness_goal, daily_calorie_goal') 
          .eq('id', user.id)
          .single();

      if (mounted) {
        setState(() {
          _username = profileData['username'] as String? ?? _username;
          _weight = (profileData['current_body_weight_kg'] as num?)?.toDouble() ?? _weight;
          _height = (profileData['height_cm'] as num?)?.toDouble() ?? _height;
          _fitnessGoal = profileData['fitness_goal'] as String? ?? _fitnessGoal; 
          _calculatedCalories = (profileData['daily_calorie_goal'] as num?)?.toDouble() ?? _calculatedCalories;

          _usernameController.text = _username;
          _weightController.text = _weight.toString();
          _heightController.text = _height.toString();
          
          _calculateCalories(); 
        });
      }
    } catch (e) {
      if (mounted) {
        print("Error loading profile from Supabase: $e");
        _usernameController.text = _username; 
        _weightController.text = _weight.toString();
        _heightController.text = _height.toString();
        _calculateCalories();
        
        if (e is PostgrestException) {
          if (e.code == 'PGRST116') { 
             ScaffoldMessenger.of(context).showSnackBar(
               const SnackBar(content: Text('Profile not yet created. Please save your details.'), backgroundColor: Colors.blue),
             );
          } else if (e.message.toLowerCase().contains('fitness_goal')) {
             ScaffoldMessenger.of(context).showSnackBar(
               const SnackBar(content: Text('DB Error: "fitness_goal" column missing. Please add it in Supabase.'), backgroundColor: Colors.red, duration: Duration(seconds: 6)),
             );
          } else if (e.message.toLowerCase().contains('daily_calorie_goal')) {
             ScaffoldMessenger.of(context).showSnackBar(
               const SnackBar(content: Text('DB Error: "daily_calorie_goal" column missing/misnamed. Check Supabase.'), backgroundColor: Colors.red, duration: Duration(seconds: 6)),
             );
          } else {
            ScaffoldMessenger.of(context).showSnackBar(
               SnackBar(content: Text('Error loading profile: ${e.message}'), backgroundColor: Colors.orange, duration: Duration(seconds: 5)),
             );
          }
        }
      }
    } finally {
       if (mounted) {
        setState(() => _loading = false);
      }
    }

    final prefs = await SharedPreferences.getInstance();
    String? imagePath = prefs.getString('profileImagePath_${user.id}');
    if (imagePath != null && mounted) {
      setState(() {
        _profileImage = File(imagePath);
      });
    }
  }

  void _calculateCalories() {
    int age = 25; 
    double bmr = (10 * _weight) + (6.25 * _height) - (5 * age) + 5; 
    double activityMultiplier = 1.2; 

    double maintenanceCalories = bmr * activityMultiplier;
    double newCalculatedCalories = maintenanceCalories;

    if (_fitnessGoal == 'bulk') {
      newCalculatedCalories = maintenanceCalories + 500;
    } else if (_fitnessGoal == 'cut') {
      newCalculatedCalories = maintenanceCalories - 500;
    }

    if (mounted) {
      setState(() {
        _calculatedCalories = newCalculatedCalories.roundToDouble();
      });
    }
  }

  Future<void> _saveProfileData() async {
    if (!_formKey.currentState!.validate()) {
      return;
    }
    if (!mounted) return;
    setState(() => _loading = true);

    final user = Supabase.instance.client.auth.currentUser;
    if (user == null) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('User not logged in. Cannot save profile.'), backgroundColor: Colors.red),
        );
        setState(() => _loading = false);
      }
      return;
    }

    _username = _usernameController.text;
    _weight = double.tryParse(_weightController.text) ?? _weight;
    _height = double.tryParse(_heightController.text) ?? _height;

    _calculateCalories(); 

    final updates = {
      'id': user.id,
      'username': _username,
      'current_body_weight_kg': _weight, 
      'height_cm': _height,
      'fitness_goal': _fitnessGoal,       
      'daily_calorie_goal': _calculatedCalories, 
      'updated_at': DateTime.now().toIso8601String(),
    };

    try {
      await Supabase.instance.client.from('user_profiles').upsert(updates);
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('Profile saved successfully!'), backgroundColor: Colors.green),
        );
      }
    } catch (e) {
      if (mounted) {
        print("Error saving profile to Supabase: $e");
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Error saving profile: ${e.toString()}'), backgroundColor: Colors.red),
        );
      }
    }

    if (_profileImage != null) {
      final prefs = await SharedPreferences.getInstance();
      await prefs.setString('profileImagePath_${user.id}', _profileImage!.path);
    }

    if (mounted) {
      setState(() => _loading = false);
    }
  }

  Future<void> _pickImage(ImageSource source) async {
    try {
      final XFile? pickedFile = await _picker.pickImage(source: source);
      if (pickedFile != null && mounted) {
        setState(() {
          _profileImage = File(pickedFile.path);
        });
        final user = Supabase.instance.client.auth.currentUser;
        if (user != null && _profileImage != null) {
            final prefs = await SharedPreferences.getInstance();
            await prefs.setString('profileImagePath_${user.id}', _profileImage!.path);
        }
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Could not select image: $e')),
        );
      }
    }
  }

  Widget _buildGoalButton(String title, String value) {
    bool isActive = _fitnessGoal == value; 
    return Expanded(
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 4.0),
        child: AnimatedContainer(
          duration: const Duration(milliseconds: 300),
          curve: Curves.easeInOut,
          child: ElevatedButton(
            style: ElevatedButton.styleFrom(
              backgroundColor: isActive 
                  ? const Color(0xFF64B5F6) 
                  : Colors.white,
              foregroundColor: isActive 
                  ? Colors.white 
                  : const Color(0xFF1976D2),
              elevation: isActive ? 8 : 2,
              shadowColor: isActive 
                  ? const Color(0xFF64B5F6).withOpacity(0.5) 
                  : Colors.grey.withOpacity(0.3),
              shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(15),
                side: BorderSide(
                  color: isActive 
                      ? const Color(0xFF64B5F6) 
                      : const Color(0xFF1976D2).withOpacity(0.3),
                  width: 2,
                ),
              ),
              padding: const EdgeInsets.symmetric(vertical: 16),
            ),
            onPressed: () {
              if (mounted) {
                setState(() {
                  _fitnessGoal = value; 
                  _calculateCalories();
                });
              }
            },
            child: Text(
              title, 
              style: TextStyle(
                fontSize: 14,
                fontWeight: isActive ? FontWeight.bold : FontWeight.w600,
              ),
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildAnimatedCard({required Widget child}) {
    return FadeTransition(
      opacity: _fadeAnimation,
      child: SlideTransition(
        position: _slideAnimation,
        child: ScaleTransition(
          scale: _scaleAnimation,
          child: Container(
            margin: const EdgeInsets.symmetric(horizontal: 20, vertical: 10),
            padding: const EdgeInsets.all(20),
            decoration: BoxDecoration(
              gradient: const LinearGradient(
                begin: Alignment.topLeft,
                end: Alignment.bottomRight,
                colors: [
                  Color(0xFFE3F2FD),
                  Color(0xFFFFFFFF),
                ],
              ),
              borderRadius: BorderRadius.circular(20),
              boxShadow: [
                BoxShadow(
                  color: const Color(0xFF64B5F6).withOpacity(0.2),
                  blurRadius: 15,
                  offset: const Offset(0, 8),
                ),
              ],
            ),
            child: child,
          ),
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Container(
        decoration: const BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topCenter,
            end: Alignment.bottomCenter,
            colors: [
              Color(0xFFE3F2FD),
              Color(0xFFF8F9FA),
            ],
          ),
        ),
        child: _loading && Supabase.instance.client.auth.currentUser != null 
            ? const Center(
                child: CircularProgressIndicator(
                  valueColor: AlwaysStoppedAnimation<Color>(Color(0xFF64B5F6)),
                ),
              )
            : SingleChildScrollView(
                child: Form(
                  key: _formKey,
                  child: Column(
                    children: <Widget>[
                      // Header Section with Gradient
                      Container(
                        padding: const EdgeInsets.only(top: 60, bottom: 40, left: 20, right: 20),
                        decoration: const BoxDecoration(
                          gradient: LinearGradient(
                            begin: Alignment.topLeft,
                            end: Alignment.bottomRight,
                            colors: [
                              Color(0xFF64B5F6),
                              Color(0xFF90CAF9),
                            ],
                          ),
                          borderRadius: BorderRadius.only(
                            bottomLeft: Radius.circular(40),
                            bottomRight: Radius.circular(40),
                          ),
                        ),
                        child: FadeTransition(
                          opacity: _fadeAnimation,
                          child: Row(
                            children: <Widget>[
                              GestureDetector(
                                onTap: () {
                                  showModalBottomSheet(
                                    context: context,
                                    backgroundColor: Colors.transparent,
                                    builder: (BuildContext bc) {
                                      return Container(
                                        decoration: const BoxDecoration(
                                          color: Colors.white,
                                          borderRadius: BorderRadius.only(
                                            topLeft: Radius.circular(25),
                                            topRight: Radius.circular(25),
                                          ),
                                        ),
                                        child: SafeArea(
                                          child: Wrap(
                                            children: <Widget>[
                                              Container(
                                                padding: const EdgeInsets.all(20),
                                                child: Column(
                                                  children: [
                                                    Container(
                                                      width: 50,
                                                      height: 5,
                                                      decoration: BoxDecoration(
                                                        color: Colors.grey[300],
                                                        borderRadius: BorderRadius.circular(10),
                                                      ),
                                                    ),
                                                    const SizedBox(height: 20),
                                                    const Text(
                                                      'Select Photo',
                                                      style: TextStyle(
                                                        fontSize: 18,
                                                        fontWeight: FontWeight.bold,
                                                        color: Color(0xFF1976D2),
                                                      ),
                                                    ),
                                                    const SizedBox(height: 20),
                                                    ListTile(
                                                      leading: Container(
                                                        padding: const EdgeInsets.all(10),
                                                        decoration: BoxDecoration(
                                                          color: const Color(0xFF64B5F6).withOpacity(0.1),
                                                          borderRadius: BorderRadius.circular(10),
                                                        ),
                                                        child: const Icon(
                                                          Icons.photo_library,
                                                          color: Color(0xFF64B5F6),
                                                        ),
                                                      ),
                                                      title: const Text(
                                                        'Gallery',
                                                        style: TextStyle(
                                                          fontWeight: FontWeight.w600,
                                                        ),
                                                      ),
                                                      onTap: () {
                                                        _pickImage(ImageSource.gallery);
                                                        Navigator.of(context).pop();
                                                      },
                                                    ),
                                                    ListTile(
                                                      leading: Container(
                                                        padding: const EdgeInsets.all(10),
                                                        decoration: BoxDecoration(
                                                          color: const Color(0xFF64B5F6).withOpacity(0.1),
                                                          borderRadius: BorderRadius.circular(10),
                                                        ),
                                                        child: const Icon(
                                                          Icons.photo_camera,
                                                          color: Color(0xFF64B5F6),
                                                        ),
                                                      ),
                                                      title: const Text(
                                                        'Camera',
                                                        style: TextStyle(
                                                          fontWeight: FontWeight.w600,
                                                        ),
                                                      ),
                                                      onTap: () {
                                                        _pickImage(ImageSource.camera);
                                                        Navigator.of(context).pop();
                                                      },
                                                    ),
                                                  ],
                                                ),
                                              ),
                                            ],
                                          ),
                                        ),
                                      );
                                    },
                                  );
                                },
                                child: Container(
                                  decoration: BoxDecoration(
                                    shape: BoxShape.circle,
                                    boxShadow: [
                                      BoxShadow(
                                        color: Colors.black.withOpacity(0.2),
                                        blurRadius: 10,
                                        offset: const Offset(0, 5),
                                      ),
                                    ],
                                  ),
                                  child: CircleAvatar(
                                    radius: 40,
                                    backgroundColor: Colors.white,
                                    backgroundImage: _profileImage != null
                                        ? FileImage(_profileImage!)
                                        : const AssetImage('assets/images/pfpcar.jpg') as ImageProvider,
                                    child: _profileImage == null 
                                        ? const Icon(
                                            Icons.camera_alt, 
                                            color: Color(0xFF64B5F6), 
                                            size: 30,
                                          ) 
                                        : null,
                                  ),
                                ),
                              ),
                              const SizedBox(width: 20),
                              Expanded(
                                child: Column(
                                  crossAxisAlignment: CrossAxisAlignment.start,
                                  children: <Widget>[
                                    Text(
                                      _username.isNotEmpty ? _username : (_userEmail ?? 'User'),
                                      style: const TextStyle(
                                        fontSize: 26,
                                        fontWeight: FontWeight.bold,
                                        color: Colors.white,
                                      ),
                                      overflow: TextOverflow.ellipsis,
                                    ),
                                    if (_userEmail != null)
                                      Text(
                                        _userEmail!,
                                        style: TextStyle(
                                          fontSize: 14,
                                          color: Colors.white.withOpacity(0.9),
                                        ),
                                        overflow: TextOverflow.ellipsis,
                                      ),
                                    const SizedBox(height: 8),
                                    Container(
                                      padding: const EdgeInsets.symmetric(
                                        horizontal: 12,
                                        vertical: 6,
                                      ),
                                      decoration: BoxDecoration(
                                        color: Colors.white.withOpacity(0.2),
                                        borderRadius: BorderRadius.circular(20),
                                      ),
                                      child: Text(
                                        '${_calculatedCalories.toStringAsFixed(0)} Cal / daily',
                                        style: const TextStyle(
                                          fontSize: 16,
                                          color: Colors.white,
                                          fontWeight: FontWeight.w600,
                                        ),
                                      ),
                                    ),
                                  ],
                                ),
                              ),
                            ],
                          ),
                        ),
                      ),
                      
                      const SizedBox(height: 20),
                      
                      // Username Field
                      _buildAnimatedCard(
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            const Text(
                              'Username',
                              style: TextStyle(
                                fontSize: 18,
                                fontWeight: FontWeight.bold,
                                color: Color(0xFF1976D2),
                              ),
                            ),
                            const SizedBox(height: 15),
                            TextFormField(
                              controller: _usernameController,
                              decoration: InputDecoration(
                                hintText: 'Enter your username',
                                filled: true,
                                fillColor: Colors.white,
                                border: OutlineInputBorder(
                                  borderRadius: BorderRadius.circular(15),
                                  borderSide: BorderSide.none,
                                ),
                                focusedBorder: OutlineInputBorder(
                                  borderRadius: BorderRadius.circular(15),
                                  borderSide: const BorderSide(
                                    color: Color(0xFF64B5F6),
                                    width: 2,
                                  ),
                                ),
                                contentPadding: const EdgeInsets.symmetric(
                                  horizontal: 20,
                                  vertical: 16,
                                ),
                              ),
                              validator: (value) {
                                if (value == null || value.isEmpty) {
                                  return 'Please enter a username';
                                }
                                return null;
                              },
                            ),
                          ],
                        ),
                      ),
                      
                      // Body Metrics
                      _buildAnimatedCard(
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            const Text(
                              'Body Metrics',
                              style: TextStyle(
                                fontSize: 18,
                                fontWeight: FontWeight.bold,
                                color: Color(0xFF1976D2),
                              ),
                            ),
                            const SizedBox(height: 20),
                            _buildInputRowWithController(
                              'Weight (kg)',
                              _weightController,
                              (value) { 
                                setState(() { 
                                  _weight = double.tryParse(value) ?? _weight;
                                  _calculateCalories();
                                });
                              },
                            ),
                            const SizedBox(height: 20),
                            _buildInputRowWithController(
                              'Height (cm)',
                              _heightController,
                              (value) {
                                setState(() {
                                  _height = double.tryParse(value) ?? _height;
                                  _calculateCalories();
                                });
                              },
                            ),
                          ],
                        ),
                      ),
                      
                      // Fitness Goal
                      _buildAnimatedCard(
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            const Text(
                              'Fitness Goal',
                              style: TextStyle(
                                fontSize: 18,
                                fontWeight: FontWeight.bold,
                                color: Color(0xFF1976D2),
                              ),
                            ),
                            const SizedBox(height: 20),
                            Row(
                              children: [
                                _buildGoalButton('Maintain', 'maintain'),
                                _buildGoalButton('Bulk', 'bulk'),
                                _buildGoalButton('Cut', 'cut'),
                              ],
                            ),
                          ],
                        ),
                      ),
                      
                      const SizedBox(height: 30),
                      
                      // Save Button
                      Padding(
                        padding: const EdgeInsets.symmetric(horizontal: 20),
                        child: Container(
                          width: double.infinity,
                          height: 60,
                          decoration: BoxDecoration(
                            gradient: const LinearGradient(
                              begin: Alignment.centerLeft,
                              end: Alignment.centerRight,
                              colors: [
                                Color(0xFF64B5F6),
                                Color(0xFF42A5F5),
                              ],
                            ),
                            borderRadius: BorderRadius.circular(20),
                            boxShadow: [
                              BoxShadow(
                                color: const Color(0xFF64B5F6).withOpacity(0.4),
                                blurRadius: 15,
                                offset: const Offset(0, 8),
                              ),
                            ],
                          ),
                          child: ElevatedButton(
                            style: ElevatedButton.styleFrom(
                              backgroundColor: Colors.transparent,
                              shadowColor: Colors.transparent,
                              shape: RoundedRectangleBorder(
                                borderRadius: BorderRadius.circular(20),
                              ),
                            ),
                            onPressed: _loading ? null : _saveProfileData,
                            child: _loading && mounted && ModalRoute.of(context)?.isCurrent == true 
                                ? const SizedBox(
                                    height: 24,
                                    width: 24,
                                    child: CircularProgressIndicator(
                                      color: Colors.white,
                                      strokeWidth: 3,
                                    ),
                                  )
                                : const Text(
                                    'Save Profile & Update Goal',
                                    style: TextStyle(
                                      fontSize: 18,
                                      fontWeight: FontWeight.bold,
                                      color: Colors.white,
                                    ),
                                  ),
                          ),
                        ),
                      ),
                      
                      const SizedBox(height: 30),
                      
                      // Logout Button
                      Padding(
                        padding: const EdgeInsets.symmetric(horizontal: 20),
                        child: Container(
                          decoration: BoxDecoration(
                            borderRadius: BorderRadius.circular(15),
                            border: Border.all(
                              color: Colors.red.withOpacity(0.3),
                              width: 1,
                            ),
                          ),
                          child: ListTile(
                            leading: Container(
                              padding: const EdgeInsets.all(8),
                              decoration: BoxDecoration(
                                color: Colors.red.withOpacity(0.1),
                                borderRadius: BorderRadius.circular(8),
                              ),
                              child: const Icon(
                                Icons.exit_to_app,
                                color: Colors.red,
                                size: 20,
                              ),
                            ),
                            title: const Text(
                              'Logout',
                              style: TextStyle(
                                color: Colors.red,
                                fontWeight: FontWeight.w600,
                              ),
                            ),
                            onTap: _logout,
                          ),
                        ),
                      ),
                      
                      const SizedBox(height: 40),
                    ],
                  ),
                ),
              ),
      ),
    );
  }

  Future<void> _logout() async {
    if (!mounted) return;
    print("DEBUG ProfileScreen: Logout button pressed.");
    try {
      await Supabase.instance.client.auth.signOut();
      print("DEBUG ProfileScreen: Supabase.auth.signOut() completed.");
    } catch (e) {
      print("DEBUG ProfileScreen: Error during Supabase.auth.signOut(): ${e.toString()}");
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Error logging out: ${e.toString()}'), backgroundColor: Colors.red),
        );
      }
    }
  }

  Widget _buildInputRowWithController(String label, TextEditingController controller, Function(String) onChanged) {
    return Row(
      mainAxisAlignment: MainAxisAlignment.spaceBetween,
      children: <Widget>[
        Text(
          label,
          style: const TextStyle(
            fontSize: 16,
            fontWeight: FontWeight.w600,
            color: Color(0xFF1976D2),
          ),
        ),
        Container(
          width: 120,
          decoration: BoxDecoration(
            borderRadius: BorderRadius.circular(12),
            boxShadow: [
              BoxShadow(
                color: const Color(0xFF64B5F6).withOpacity(0.1),
                blurRadius: 8,
                offset: const Offset(0, 2),
              ),
            ],
          ),
          child: TextFormField(
            controller: controller,
            keyboardType: const TextInputType.numberWithOptions(decimal: true),
            textAlign: TextAlign.center,
            style: const TextStyle(
              fontSize: 16,
              fontWeight: FontWeight.w600,
            ),
            decoration: InputDecoration(
              filled: true,
              fillColor: Colors.white,
              contentPadding: const EdgeInsets.symmetric(
                vertical: 12,
                horizontal: 16,
              ),
              border: OutlineInputBorder(
                borderRadius: BorderRadius.circular(12),
                borderSide: BorderSide.none,
              ),
              focusedBorder: OutlineInputBorder(
                borderRadius: BorderRadius.circular(12),
                borderSide: const BorderSide(
                  color: Color(0xFF64B5F6),
                  width: 2,
                ),
              ),
            ),
            validator: (value) {
              if (value == null || value.isEmpty) {
                return 'Required';
              }
              if (double.tryParse(value) == null) {
                return 'Invalid';
              }
              return null;
            },
            onChanged: onChanged,
          ),
        ),
      ],
    );
  }
}