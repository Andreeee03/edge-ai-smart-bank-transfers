import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const BenchmarkApp());
}

class BenchmarkPrompt {
  final String id;
  final String task;
  final String prompt;

  const BenchmarkPrompt({
    required this.id,
    required this.task,
    required this.prompt,
  });
}

const benchmarkPrompts = <BenchmarkPrompt>[
  BenchmarkPrompt(
    id: 'P1',
    task: 'GENERATION',
    prompt:
        'Generate exactly two concise and natural '
        'bank-transfer descriptions using only the '
        'information provided.\n'
        'Return two alternative descriptions without '
        'adding unsupported information.\n'
        'Category: RENT\n'
        'Beneficiary: Green Residence\n'
        'Amount: 850 EUR\n'
        'Reference period: September 2026\n\n',
  ),

  BenchmarkPrompt(
    id: 'P2',
    task: 'GENERATION_CALENDAR',
    prompt:
        'Generate exactly two concise and natural '
        'bank-transfer descriptions using only the '
        'information provided.\n'
        'Return two alternative descriptions without '
        'adding unsupported information.\n'
        'Category: HEALTH\n'
        'Beneficiary: Dental Clinic Milano\n'
        'Amount: 120 EUR\n'
        'Reference period: 15 September 2026\n'
        'Calendar context:\n'
        '- Event: Dentist appointment\n'
        '- Date: 2026-09-15\n'
        '- Event category: Health\n\n',
  ),

  BenchmarkPrompt(
    id: 'P3',
    task: 'COMPLETION',
    prompt:
        'Complete the following partially written '
        'bank-transfer description.\n'
        'Generate exactly two concise and natural '
        'completed alternatives using only the '
        'information provided.\n'
        'Category: UTILITIES\n'
        'Beneficiary: Energy Service\n'
        'Amount: 94.50 EUR\n'
        'Reference period: August 2026\n'
        'Partial description: Electricity bill for\n\n',
  ),

  BenchmarkPrompt(
    id: 'P4',
    task: 'COMPLETION',
    prompt:
        'Complete the following partially written '
        'bank-transfer description.\n'
        'Generate exactly two concise and natural '
        'completed alternatives using only the '
        'information provided.\n'
        'Category: EDUCATION\n'
        'Beneficiary: University Services\n'
        'Amount: 350 EUR\n'
        'Reference period: academic year 2026\n'
        'Partial description: University fee payment\n\n',
  ),

  BenchmarkPrompt(
    id: 'P5',
    task: 'NORMALIZATION',
    prompt:
        'Normalize the following bank-transfer '
        'description by making it clear, concise '
        'and natural.\n'
        'Generate exactly two alternative normalized '
        'descriptions while preserving the original '
        'meaning and without adding unsupported '
        'information.\n'
        'Category: REIMBURSEMENT\n'
        'Beneficiary: Marco Rossi\n'
        'Amount: 48 EUR\n'
        'Original description: money back marco dinner last saturday\n\n',
  ),

  BenchmarkPrompt(
    id: 'P6',
    task: 'NORMALIZATION',
    prompt:
        'Normalize the following bank-transfer '
        'description by making it clear, concise '
        'and natural.\n'
        'Generate exactly two alternative normalized '
        'descriptions while preserving the original '
        'meaning and without adding unsupported '
        'information.\n'
        'Category: VEHICLE SERVICE\n'
        'Beneficiary: Auto Service Center\n'
        'Amount: 215 EUR\n'
        'Original description: payment car service oil and filters\n\n',
  ),
];

class BenchmarkApp extends StatelessWidget {
  const BenchmarkApp({super.key});

  @override
  Widget build(BuildContext context) {
    return const MaterialApp(
      debugShowCheckedModeBanner: false,
      home: BenchmarkPage(),
    );
  }
}

class BenchmarkPage extends StatefulWidget {
  const BenchmarkPage({super.key});

  @override
  State<BenchmarkPage> createState() => _BenchmarkPageState();
}

class _BenchmarkPageState extends State<BenchmarkPage> {
  static const platform = MethodChannel('edge_ai/native');

  static const int cycles =
      int.fromEnvironment('BENCHMARK_CYCLES', defaultValue: 17);

  String status = 'Preparing benchmark...';
  int completed = 0;
  bool started = false;

  int get totalRuns => benchmarkPrompts.length * cycles;

  @override
  void initState() {
    super.initState();

    WidgetsBinding.instance.addPostFrameCallback((_) {
      _runBenchmark();
    });
  }

  Future<void> _runBenchmark() async {
    if (started) {
      return;
    }

    started = true;

    try {
      debugPrint(
        'DEPLOYMENT_BENCHMARK_START '
        'cycles=$cycles '
        'prompts=${benchmarkPrompts.length} '
        'total=$totalRuns',
      );

      setState(() {
        status = 'Loading model...';
      });

      final loadWatch = Stopwatch()..start();

      final loadResult =
          await platform.invokeMethod<String>('loadModel');

      loadWatch.stop();

      debugPrint(
        'DEPLOYMENT_BENCHMARK_MODEL '
        'model_load_ms=${loadWatch.elapsedMicroseconds / 1000.0} '
        'result="$loadResult"',
      );

      setState(() {
        status = 'Creating inference context...';
      });

      final contextWatch = Stopwatch()..start();

      final contextResult =
          await platform.invokeMethod<String>('createContext');

      contextWatch.stop();

      debugPrint(
        'DEPLOYMENT_BENCHMARK_CONTEXT '
        'context_creation_ms=${contextWatch.elapsedMicroseconds / 1000.0} '
        'result="$contextResult"',
      );

      var run = 0;

      for (var cycle = 1; cycle <= cycles; cycle++) {
        for (final benchmarkPrompt in benchmarkPrompts) {
          run++;

          if (mounted) {
            setState(() {
              completed = run;
              status =
                  'Run $run / $totalRuns\n'
                  '${benchmarkPrompt.id} - '
                  '${benchmarkPrompt.task}';
            });
          }

          debugPrint(
            'DEPLOYMENT_BENCHMARK_RUN_START '
            'run=$run '
            'cycle=$cycle '
            'prompt=${benchmarkPrompt.id} '
            'task=${benchmarkPrompt.task}',
          );

          final result =
              await platform.invokeMethod<String>(
            'generate',
            {
              'prompt': benchmarkPrompt.prompt,
            },
          );

          if (result == null ||
              !result.startsWith('Generation OK')) {
            throw Exception(
              'Generation failed at run $run '
              '(${benchmarkPrompt.id}).',
            );
          }

          debugPrint(
            'DEPLOYMENT_BENCHMARK_RUN_END '
            'run=$run '
            'cycle=$cycle '
            'prompt=${benchmarkPrompt.id} '
            'task=${benchmarkPrompt.task}',
          );
        }
      }

      debugPrint(
        'DEPLOYMENT_BENCHMARK_COMPLETE '
        'runs=$completed',
      );

      if (mounted) {
        setState(() {
          status =
              'Benchmark completed successfully.\n'
              '$completed / $totalRuns runs';
        });
      }
    } catch (e, stackTrace) {
      debugPrint(
        'DEPLOYMENT_BENCHMARK_ERROR '
        '$e\n$stackTrace',
      );

      if (mounted) {
        setState(() {
          status =
              'Benchmark failed after '
              '$completed / $totalRuns runs.\n'
              '$e';
        });
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    final progress =
        totalRuns == 0 ? 0.0 : completed / totalRuns;

    return Scaffold(
      appBar: AppBar(
        title: const Text('Deployment Benchmark'),
      ),
      body: Padding(
        padding: const EdgeInsets.all(24),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            const Icon(
              Icons.speed,
              size: 64,
            ),
            const SizedBox(height: 24),
            Text(
              status,
              textAlign: TextAlign.center,
              style: Theme.of(context).textTheme.titleMedium,
            ),
            const SizedBox(height: 24),
            LinearProgressIndicator(
              value: progress,
            ),
            const SizedBox(height: 12),
            Text(
              '$completed / $totalRuns',
              textAlign: TextAlign.center,
            ),
          ],
        ),
      ),
    );
  }
}
